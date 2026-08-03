# -*- coding: utf-8 -*-
#1. 中国国债零息收益率曲线 -> 固定期限远期利率；
#2. 现实概率测度 P 与风险中性测度 Q 下的状态动态；
#3. 影子短端 s_t = delta0 + delta1' X_t；
#4. 有效利率下限进入全部期限的 Wu–Xia 非线性定价；
#5. 扩展 Kalman 滤波与 RTS 平滑；
#6. FixParam / MLE / RollingMLE；
#7. 输出影子利率、名义短端、下限约束概率、置信区间及拟合诊断。

#说明：
#- Nelson–Siegel 仅用于参数与状态初始化，不参与最终 Wu–Xia 定价。
#- 默认 MLE 使用稳定性较高的对角 AP/AQ、Cholesky Sigma 和对角 R。
#- CurveControls 仅作为可选测量修正项，不改变核心影子短端定义。
#"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence
import copy
import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

Array = np.ndarray


# -----------------------------------------------------------------------------
# 配置与参数对象
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LBSpec:
    """有效利率下限设定。

    mode:
        "Fixed"       : 全样本固定 value；
        "RegimeFixed" : 按 regime_series 使用 regime_values；
        "Estimated"   : MLE 估计一个或多个分阶段下限。
    """

    mode: str = "Fixed"
    value: float = 0.0
    regime_series: Sequence[int] | None = None
    regime_values: Sequence[float] | None = None
    bounds: tuple[float, float] = (-0.03, 0.06)
    prior_mean: float | Sequence[float] | None = None
    prior_sd: float | Sequence[float] | None = None


@dataclass(frozen=True)
class ModelSpec:
    """模型结构与数值设置。"""

    state_dim: int = 3
    state_step: float = 1.0 / 12.0
    variance_tolerance: float = 1e-12
    numerical_jitter: float = 1e-9
    interpolation_penalty: float = 4.0
    diagonal_dynamics: bool = True
    diagonal_measurement: bool = True
    estimate_delta: bool = True
    curve_control_mode: str = "Off"  # "Off" / "ObservedControls"
    control_names: tuple[str, ...] = ()
    estimate_controls: bool = False
    control_ridge: float = 1e-4
    num_starts: int = 3
    maxiter: int = 500
    optimizer_ftol: float = 1e-8
    ci_level: float = 0.95
    binding_probability_threshold: float = 0.5
    rolling_window: int | None = None
    rolling_step: int = 1
    random_seed: int = 20260803
    ns_lambda_init: float = 0.7


@dataclass
class WXParams:
    """Wu–Xia 模型参数。"""

    cP: Array
    AP: Array
    cQ: Array
    AQ: Array
    Sigma: Array
    delta0: float
    delta1: Array
    R: Array
    lb_params: Array = field(default_factory=lambda: np.empty(0, dtype=float))
    Gamma: Array | None = None

    def copy(self) -> "WXParams":
        return WXParams(
            cP=np.array(self.cP, float, copy=True),
            AP=np.array(self.AP, float, copy=True),
            cQ=np.array(self.cQ, float, copy=True),
            AQ=np.array(self.AQ, float, copy=True),
            Sigma=np.array(self.Sigma, float, copy=True),
            delta0=float(self.delta0),
            delta1=np.array(self.delta1, float, copy=True),
            R=np.array(self.R, float, copy=True),
            lb_params=np.array(self.lb_params, float, copy=True),
            Gamma=None if self.Gamma is None else np.array(self.Gamma, float, copy=True),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cP": self.cP,
            "AP": self.AP,
            "cQ": self.cQ,
            "AQ": self.AQ,
            "Sigma": self.Sigma,
            "Omega": self.Sigma @ self.Sigma.T,
            "Delta0": self.delta0,
            "Delta1": self.delta1,
            "R": self.R,
            "LBParam": self.lb_params,
            "Gamma": self.Gamma,
        }


# -----------------------------------------------------------------------------
# 通用数值工具
# -----------------------------------------------------------------------------


def _symmetrize(a: Array) -> Array:
    return 0.5 * (a + a.T)


def _nearest_psd(a: Array, floor: float = 1e-12) -> Array:
    a = _symmetrize(np.asarray(a, float))
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, floor)
    return _symmetrize((vecs * vals) @ vecs.T)


def _chol_with_jitter(a: Array, jitter: float) -> tuple[Array, Array]:
    """返回 Cholesky 分解及加入微扰后的矩阵。"""

    base = _symmetrize(np.asarray(a, float))
    eye = np.eye(base.shape[0])
    eps = max(float(jitter), 1e-14)
    for _ in range(10):
        adjusted = base + eps * eye
        try:
            return np.linalg.cholesky(adjusted), adjusted
        except np.linalg.LinAlgError:
            eps *= 10.0
    adjusted = _nearest_psd(base, floor=eps)
    return np.linalg.cholesky(adjusted), adjusted


def _solve_spd_from_chol(chol: Array, b: Array) -> Array:
    y = np.linalg.solve(chol, b)
    return np.linalg.solve(chol.T, y)


def _stable_logdet_from_chol(chol: Array) -> float:
    return float(2.0 * np.sum(np.log(np.diag(chol))))


def _as_2d_float(a: Any, name: str) -> Array:
    out = np.asarray(a, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return out


def _as_1d_float(a: Any, name: str) -> Array:
    out = np.asarray(a, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return out


def _spectral_radius(a: Array) -> float:
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def _check_square(a: Array, n: int, name: str) -> None:
    if a.shape != (n, n):
        raise ValueError(f"{name} must have shape {(n, n)}, got {a.shape}")


def _normalize_regime_series(
    regime_series: Sequence[int] | None,
    t_len: int,
) -> tuple[Array | None, int]:
    if regime_series is None:
        return None, 0
    raw = np.asarray(regime_series)
    if raw.ndim != 1 or len(raw) != t_len:
        raise ValueError("LB regime_series must be one-dimensional with length T")
    if not np.all(np.isfinite(raw.astype(float))):
        raise ValueError("LB regime_series contains missing values")
    unique = list(dict.fromkeys(raw.tolist()))
    mapping = {value: i for i, value in enumerate(unique)}
    encoded = np.asarray([mapping[v] for v in raw.tolist()], dtype=int)
    return encoded, len(unique)


# -----------------------------------------------------------------------------
# 零息曲线与固定期限远期利率
# -----------------------------------------------------------------------------


def interpolate_zero_curve(
    zero_yield_row: Array,
    curve_maturities: Array,
) -> tuple[Array, Array]:
    """对单期零息曲线插值。

    返回：
        filled_curve: 插值后的曲线；若全期缺失则全部为 NaN；
        source_missing: 原始缺失位置标志。
    """

    row = np.asarray(zero_yield_row, float).copy()
    mats = np.asarray(curve_maturities, float)
    source_missing = ~np.isfinite(row)
    ok = ~source_missing

    if np.sum(ok) == 0:
        return np.full_like(row, np.nan), source_missing
    if np.sum(ok) == 1:
        row[source_missing] = row[ok][0]
        return row, source_missing

    row[source_missing] = np.interp(
        mats[source_missing],
        mats[ok],
        row[ok],
    )
    return row, source_missing


def _interp_value_and_flag(
    target: float,
    mats: Array,
    filled_row: Array,
    original_row: Array,
    atol: float = 1e-10,
) -> tuple[float, bool]:
    """在曲线上读取 target，并判断是否依赖插值或外推。"""

    if target < mats[0] - atol or target > mats[-1] + atol:
        return np.nan, True

    exact = np.where(np.isclose(mats, target, atol=atol, rtol=0.0))[0]
    if len(exact) > 0:
        j = int(exact[0])
        return float(filled_row[j]), not np.isfinite(original_row[j])

    value = float(np.interp(target, mats, filled_row))
    return value, True


def build_forward_observations(
    zero_yields: Array,
    curve_maturities: Sequence[float],
    forward_starts: Sequence[float],
    forward_tenor: float,
    missing_rule: str = "CurveInterpolate",
) -> tuple[Array, Array, Array]:
    """由连续复利零息收益率构造固定期限远期利率。

    Parameters
    ----------
    zero_yields:
        T x K 的零息收益率，单位为年化小数。
    curve_maturities:
        零息曲线期限网格，单位为年。
    forward_starts:
        远期利率起始期限集合。
    forward_tenor:
        远期利率自身期限，例如 1/12 表示 1 个月。
    missing_rule:
        "CurveInterpolate"：在零息曲线层面插值，并记录插值标志；
        "DropMaturity"：只有起点和终点均为原始有效观测时才保留。

    Returns
    -------
    forward_obs:
        T x M 固定期限远期利率。
    interpolation_flags:
        T x M，True 表示该远期点依赖插值。
    zero_filled:
        插值后的零息收益率曲线。
    """

    y = _as_2d_float(zero_yields, "zero_yields")
    mats = _as_1d_float(curve_maturities, "curve_maturities")
    starts = _as_1d_float(forward_starts, "forward_starts")

    if y.shape[1] != len(mats):
        raise ValueError("zero_yields maturity dimension mismatch")
    if np.any(np.diff(mats) <= 0):
        raise ValueError("curve_maturities must be strictly increasing")
    if forward_tenor <= 0:
        raise ValueError("forward_tenor must be positive")

    t_len = y.shape[0]
    m_len = len(starts)
    forward = np.full((t_len, m_len), np.nan)
    flags = np.ones((t_len, m_len), dtype=bool)
    zero_filled = np.full_like(y, np.nan)
    rule = missing_rule.strip().lower()

    if rule not in {"curveinterpolate", "interpolate", "dropmaturity"}:
        raise ValueError("missing_rule must be CurveInterpolate or DropMaturity")

    for t in range(t_len):
        filled, _ = interpolate_zero_curve(y[t], mats)
        zero_filled[t] = filled
        if not np.any(np.isfinite(filled)):
            continue

        for j, start in enumerate(starts):
            end = float(start + forward_tenor)
            r0, flag0 = _interp_value_and_flag(start, mats, filled, y[t])
            r1, flag1 = _interp_value_and_flag(end, mats, filled, y[t])
            interpolation_used = bool(flag0 or flag1)

            if not np.isfinite(r0) or not np.isfinite(r1):
                continue
            if rule == "dropmaturity" and interpolation_used:
                continue

            log_p0 = -float(start) * r0
            log_p1 = -end * r1
            forward[t, j] = -(log_p1 - log_p0) / forward_tenor
            flags[t, j] = interpolation_used

    return forward, flags, zero_filled


# -----------------------------------------------------------------------------
# Nelson–Siegel：仅用于初始状态与初始参数
# -----------------------------------------------------------------------------


def nelson_siegel_initial_loadings(maturities: Array, lam: float) -> Array:
    m = np.asarray(maturities, float)
    z = np.maximum(lam * m, 1e-12)
    l1 = (1.0 - np.exp(-z)) / z
    l2 = l1 - np.exp(-z)
    return np.column_stack([np.ones(len(m)), l1, l2])


def _fill_matrix_for_initialization(y: Array, x_axis: Array) -> Array:
    out = np.asarray(y, float).copy()
    t_len = out.shape[0]

    for t in range(t_len):
        row = out[t]
        ok = np.isfinite(row)
        if np.sum(ok) >= 2:
            row[~ok] = np.interp(x_axis[~ok], x_axis[ok], row[ok])
        elif np.sum(ok) == 1:
            row[~ok] = row[ok][0]
        elif t > 0 and np.all(np.isfinite(out[t - 1])):
            row[:] = out[t - 1]
        out[t] = row

    col_means = np.nanmean(out, axis=0)
    overall = float(np.nanmean(out)) if np.any(np.isfinite(out)) else 0.0
    col_means = np.where(np.isfinite(col_means), col_means, overall)
    inds = np.where(~np.isfinite(out))
    out[inds] = col_means[inds[1]]
    return out


def _estimate_var1(factors: Array, diagonal: bool) -> tuple[Array, Array, Array]:
    n = factors.shape[1]
    if len(factors) < 3:
        return np.zeros(n), 0.95 * np.eye(n), 1e-4 * np.eye(n)

    x = factors[:-1]
    y = factors[1:]
    z = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(z, y, rcond=None)
    c = beta[0]
    a = beta[1:].T

    if diagonal:
        a = np.diag(np.clip(np.diag(a), -0.98, 0.98))
    else:
        rho = _spectral_radius(a)
        if rho >= 0.98:
            a = a * (0.98 / max(rho, 1e-12))

    resid = y - (c + x @ a.T)
    omega = np.cov(resid.T) if len(resid) > 1 else 1e-4 * np.eye(n)
    omega = _nearest_psd(np.atleast_2d(omega), floor=1e-8)
    return np.asarray(c, float), np.asarray(a, float), omega


def initialize_parameters(
    forward_obs: Array,
    forward_starts: Array,
    model_spec: ModelSpec,
    lb_spec: LBSpec,
    controls: Array | None,
    param0: Mapping[str, Any] | WXParams | None,
) -> tuple[WXParams, Array, Array]:
    """以 NS/PCA 风格因子初始化 P/Q 参数与状态。"""

    if isinstance(param0, WXParams):
        p = param0.copy()
        n = model_spec.state_dim
        x0 = np.zeros(n)
        p0 = np.eye(n)
        return p, x0, p0

    n = model_spec.state_dim
    m = forward_obs.shape[1]
    if n != 3:
        raise ValueError(
            "The current initializer supports state_dim=3. "
            "Provide param0 and init_state for other dimensions."
        )
    if m < 3:
        raise ValueError("At least three forward maturities are required")

    f_fill = _fill_matrix_for_initialization(forward_obs, forward_starts)
    h0 = nelson_siegel_initial_loadings(
        forward_starts,
        model_spec.ns_lambda_init,
    )
    factors = np.asarray(
        [np.linalg.lstsq(h0, row, rcond=None)[0] for row in f_fill],
        float,
    )

    c_p, a_p, omega = _estimate_var1(
        factors,
        diagonal=model_spec.diagonal_dynamics,
    )
    sigma = np.linalg.cholesky(_nearest_psd(omega, floor=1e-8))

    fitted = factors @ h0.T
    resid = f_fill - fitted
    rdiag = np.maximum(np.nanvar(resid, axis=0), 1e-7)
    r = np.diag(rdiag)

    delta1 = np.array([1.0, 1.0, 0.0], dtype=float)
    short_fit = factors @ delta1
    delta0 = float(np.nanmean(f_fill[:, 0] - short_fit))

    regime, regime_count = _normalize_regime_series(
        lb_spec.regime_series,
        len(forward_obs),
    )
    mode = lb_spec.mode.strip().lower()
    if mode == "estimated":
        count = regime_count if regime is not None else 1
        if lb_spec.prior_mean is None:
            lb_params = np.full(count, lb_spec.value, dtype=float)
        else:
            prior = np.asarray(lb_spec.prior_mean, float)
            lb_params = (
                np.full(count, float(prior))
                if prior.ndim == 0
                else np.broadcast_to(prior, (count,)).copy()
            )
    elif mode == "regimefixed":
        if lb_spec.regime_values is None:
            raise ValueError("RegimeFixed requires regime_values")
        lb_params = np.asarray(lb_spec.regime_values, float)
    else:
        lb_params = np.array([lb_spec.value], dtype=float)

    gamma = None
    if model_spec.curve_control_mode.lower() == "observedcontrols":
        cnum = 0 if controls is None else controls.shape[1]
        gamma = np.zeros((m, cnum), dtype=float)

    p = WXParams(
        cP=c_p,
        AP=a_p,
        cQ=c_p.copy(),
        AQ=a_p.copy(),
        Sigma=sigma,
        delta0=delta0,
        delta1=delta1,
        R=r,
        lb_params=lb_params,
        Gamma=gamma,
    )

    if param0 is not None:
        raw = dict(param0)
        p.cP = np.asarray(raw.get("cP", p.cP), float)
        p.AP = np.asarray(raw.get("AP", raw.get("A", p.AP)), float)
        p.cQ = np.asarray(raw.get("cQ", p.cQ), float)
        p.AQ = np.asarray(raw.get("AQ", p.AQ), float)
        p.Sigma = np.asarray(raw.get("Sigma", p.Sigma), float)
        p.delta0 = float(raw.get("Delta0", raw.get("delta0", p.delta0)))
        p.delta1 = np.asarray(raw.get("Delta1", raw.get("delta1", p.delta1)), float)
        p.R = np.asarray(raw.get("R", p.R), float)
        if "LBParam" in raw:
            p.lb_params = np.atleast_1d(np.asarray(raw["LBParam"], float))
        if "Gamma" in raw and raw["Gamma"] is not None:
            p.Gamma = np.asarray(raw["Gamma"], float)

    p0 = np.cov(factors.T) if len(factors) > 1 else np.eye(n)
    p0 = _nearest_psd(np.atleast_2d(p0), floor=1e-6)
    return p, factors[0], p0


# -----------------------------------------------------------------------------
# P/Q 状态动态、Wu–Xia 定价及 Jacobian
# -----------------------------------------------------------------------------


def normal_pdf(z: Array | float) -> Array | float:
    z_arr = np.asarray(z, float)
    out = np.exp(-0.5 * z_arr * z_arr) / math.sqrt(2.0 * math.pi)
    return float(out) if out.ndim == 0 else out


def normal_cdf(z: Array | float) -> Array | float:
    out = ndtr(np.asarray(z, float))
    return float(out) if np.asarray(out).ndim == 0 else out


def wuxia_g(z: Array | float) -> Array | float:
    z_arr = np.asarray(z, float)
    out = z_arr * ndtr(z_arr) + normal_pdf(z_arr)
    return float(out) if out.ndim == 0 else out


def steps_from_maturity(
    maturity: float,
    state_step: float,
    tolerance: float = 1e-8,
) -> int:
    raw = float(maturity) / float(state_step)
    rounded = int(round(raw))
    if abs(raw - rounded) > tolerance:
        raise ValueError(
            f"Maturity {maturity} is not aligned with state_step {state_step}"
        )
    if rounded < 0:
        raise ValueError("Maturity must be nonnegative")
    return rounded


def _matrix_geometric_sum(a: Array, k: int) -> Array:
    n = a.shape[0]
    total = np.zeros((n, n))
    power = np.eye(n)
    for _ in range(k):
        total += power
        power = power @ a
    return total


def gaussian_forward_coefficients(
    k: int,
    params: WXParams,
) -> tuple[float, Array, float]:
    """计算未受下限约束的高斯仿射远期利率系数。

    返回 a_k、b_k、sigma_k，使：
        GaussianForward_k(X_t) = a_k + b_k' X_t

    a_k 按附录 A3 纳入 -1/2 方差风险调整项。
    """

    n = len(params.cQ)
    if k == 0:
        return float(params.delta0), np.asarray(params.delta1, float), 0.0

    aq_k = np.linalg.matrix_power(params.AQ, k)
    gk = _matrix_geometric_sum(params.AQ, k)
    omega = params.Sigma @ params.Sigma.T

    b_k = aq_k.T @ params.delta1
    risk_adjustment = 0.5 * float(
        params.delta1 @ gk @ omega @ gk.T @ params.delta1
    )
    a_k = float(
        params.delta0
        + params.delta1 @ (gk @ params.cQ)
        - risk_adjustment
    )

    var_x = np.zeros((n, n))
    power = np.eye(n)
    for _ in range(k):
        var_x += power @ omega @ power.T
        power = power @ params.AQ
    sigma2 = float(params.delta1 @ var_x @ params.delta1)
    sigma_k = math.sqrt(max(sigma2, 0.0))
    return a_k, b_k, sigma_k


def wuxia_forward_rate_and_jacobian(
    x: Array,
    lb: float,
    start_step: int,
    tenor_steps: int,
    params: WXParams,
    variance_tolerance: float,
) -> tuple[float, Array]:
    """计算固定期限远期利率及其对当前状态的 Jacobian。"""

    n = len(x)
    total_rate = 0.0
    total_jac = np.zeros(n)

    for offset in range(tenor_steps):
        k = start_step + offset
        a_k, b_k, sigma_k = gaussian_forward_coefficients(k, params)
        mu = float(a_k + b_k @ x)

        if sigma_k <= variance_tolerance:
            if mu > lb:
                rate = mu
                jac = b_k
            else:
                rate = lb
                jac = np.zeros(n)
        else:
            z = (mu - lb) / sigma_k
            rate = float(lb + sigma_k * wuxia_g(z))
            jac = float(normal_cdf(z)) * b_k

        total_rate += rate
        total_jac += jac

    scale = 1.0 / tenor_steps
    return total_rate * scale, total_jac * scale


def build_control_matrix(
    control_series: Mapping[str, Sequence[float]] | None,
    model_spec: ModelSpec,
    t_len: int,
) -> Array | None:
    if model_spec.curve_control_mode.lower() != "observedcontrols":
        return None
    if not model_spec.control_names:
        return np.empty((t_len, 0), dtype=float)
    if control_series is None:
        raise ValueError("ObservedControls requires control_series")

    cols = []
    for name in model_spec.control_names:
        if name not in control_series:
            raise KeyError(f"Missing control series: {name}")
        arr = np.asarray(control_series[name], float)
        if arr.ndim != 1 or len(arr) < t_len:
            raise ValueError(f"Control series {name} must have at least T observations")
        cols.append(arr[:t_len])
    return np.column_stack(cols)


def _lb_path(
    params: WXParams,
    lb_spec: LBSpec,
    t_len: int,
) -> Array:
    mode = lb_spec.mode.strip().lower()
    regime, regime_count = _normalize_regime_series(lb_spec.regime_series, t_len)

    if mode == "fixed":
        return np.full(t_len, float(lb_spec.value))
    if mode == "regimefixed":
        if regime is None or lb_spec.regime_values is None:
            raise ValueError("RegimeFixed requires regime_series and regime_values")
        values = np.asarray(lb_spec.regime_values, float)
        if len(values) != regime_count:
            raise ValueError("regime_values length does not match number of regimes")
        return values[regime]
    if mode == "estimated":
        if regime is None:
            if len(params.lb_params) != 1:
                raise ValueError("Estimated fixed LB requires one lb parameter")
            return np.full(t_len, float(params.lb_params[0]))
        if len(params.lb_params) != regime_count:
            raise ValueError("Estimated regime LB parameter count mismatch")
        return params.lb_params[regime]
    raise ValueError(f"Unsupported LB mode: {lb_spec.mode}")


def observation_vector_and_jacobian(
    t: int,
    x: Array,
    lb: float,
    indices: Array,
    forward_starts: Array,
    tenor_steps: int,
    params: WXParams,
    model_spec: ModelSpec,
    controls: Array | None,
) -> tuple[Array, Array]:
    values = np.empty(len(indices), dtype=float)
    jac = np.empty((len(indices), len(x)), dtype=float)

    for q, j in enumerate(indices):
        start_step = steps_from_maturity(
            float(forward_starts[j]),
            model_spec.state_step,
        )
        rate, row_jac = wuxia_forward_rate_and_jacobian(
            x=x,
            lb=lb,
            start_step=start_step,
            tenor_steps=tenor_steps,
            params=params,
            variance_tolerance=model_spec.variance_tolerance,
        )

        if (
            controls is not None
            and params.Gamma is not None
            and params.Gamma.shape[1] > 0
        ):
            control_row = controls[t]
            if np.all(np.isfinite(control_row)):
                rate += float(params.Gamma[j] @ control_row)

        values[q] = rate
        jac[q] = row_jac

    return values, jac


# -----------------------------------------------------------------------------
# 扩展 Kalman 滤波与 RTS 平滑
# -----------------------------------------------------------------------------


def ekf_wuxia(
    forward_obs: Array,
    interpolation_flags: Array,
    forward_starts: Array,
    forward_tenor: float,
    params: WXParams,
    x0: Array,
    p0: Array,
    lb_spec: LBSpec,
    model_spec: ModelSpec,
    controls: Array | None = None,
) -> dict[str, Any]:
    f = _as_2d_float(forward_obs, "forward_obs")
    flags = np.asarray(interpolation_flags, bool)
    if flags.shape != f.shape:
        raise ValueError("interpolation_flags shape mismatch")

    t_len, m_len = f.shape
    n = len(x0)
    _check_square(params.AP, n, "AP")
    _check_square(params.AQ, n, "AQ")
    _check_square(params.Sigma, n, "Sigma")
    if params.R.shape != (m_len, m_len):
        raise ValueError("R shape mismatch")

    tenor_steps = steps_from_maturity(
        forward_tenor,
        model_spec.state_step,
    )
    if tenor_steps <= 0:
        raise ValueError("forward_tenor must contain at least one state step")

    lb_path = _lb_path(params, lb_spec, t_len)
    omega = params.Sigma @ params.Sigma.T

    x = np.asarray(x0, float).copy()
    p = _nearest_psd(np.asarray(p0, float), floor=model_spec.numerical_jitter)

    x_pred = np.zeros((t_len, n))
    p_pred = np.zeros((t_len, n, n))
    x_filt = np.zeros((t_len, n))
    p_filt = np.zeros((t_len, n, n))
    innovation_full = np.full((t_len, m_len), np.nan)
    fitted_full = np.full((t_len, m_len), np.nan)
    observed_mask = np.zeros((t_len, m_len), dtype=bool)
    ll_contrib = np.zeros(t_len)
    loglik = 0.0

    for t in range(t_len):
        xp = params.cP + params.AP @ x
        pp = _symmetrize(params.AP @ p @ params.AP.T + omega)

        x_pred[t] = xp
        p_pred[t] = pp

        idx = np.where(np.isfinite(f[t]))[0]
        if len(idx) == 0:
            x, p = xp, pp
            x_filt[t], p_filt[t] = x, p
            continue

        y_now = f[t, idx]
        f_pred, h_jac = observation_vector_and_jacobian(
            t=t,
            x=xp,
            lb=float(lb_path[t]),
            indices=idx,
            forward_starts=forward_starts,
            tenor_steps=tenor_steps,
            params=params,
            model_spec=model_spec,
            controls=controls,
        )

        r_now = params.R[np.ix_(idx, idx)].copy()
        for q, j in enumerate(idx):
            if flags[t, j]:
                r_now[q, q] *= model_spec.interpolation_penalty

        innovation = y_now - f_pred
        s = _symmetrize(h_jac @ pp @ h_jac.T + r_now)
        chol_s, s_adjusted = _chol_with_jitter(s, model_spec.numerical_jitter)

        ph_t = pp @ h_jac.T
        k_gain = _solve_spd_from_chol(chol_s, ph_t.T).T
        x = xp + k_gain @ innovation

        i_n = np.eye(n)
        left = i_n - k_gain @ h_jac
        p = _symmetrize(left @ pp @ left.T + k_gain @ r_now @ k_gain.T)
        p = _nearest_psd(p, floor=model_spec.numerical_jitter)

        quad = float(innovation @ _solve_spd_from_chol(chol_s, innovation))
        ll_t = -0.5 * (
            _stable_logdet_from_chol(chol_s)
            + quad
            + len(innovation) * math.log(2.0 * math.pi)
        )
        loglik += ll_t
        ll_contrib[t] = ll_t

        x_filt[t] = x
        p_filt[t] = p
        innovation_full[t, idx] = innovation
        fitted_full[t, idx] = f_pred
        observed_mask[t, idx] = True

    return {
        "x_pred": x_pred,
        "P_pred": p_pred,
        "x_filt": x_filt,
        "P_filt": p_filt,
        "loglik": float(loglik),
        "ll_contrib": ll_contrib,
        "innovations": innovation_full,
        "forward_fit_filter": fitted_full,
        "observed_mask": observed_mask,
        "LBPath": lb_path,
    }


def rts_smoother_affine(
    filter_out: Mapping[str, Any],
    ap: Array,
    numerical_jitter: float,
) -> tuple[Array, Array]:
    xf = np.asarray(filter_out["x_filt"], float)
    pf = np.asarray(filter_out["P_filt"], float)
    xp = np.asarray(filter_out["x_pred"], float)
    pp = np.asarray(filter_out["P_pred"], float)

    t_len, n = xf.shape
    xs = xf.copy()
    ps = pf.copy()

    for t in range(t_len - 2, -1, -1):
        chol, _ = _chol_with_jitter(pp[t + 1], numerical_jitter)
        right = pf[t] @ ap.T
        j_gain = _solve_spd_from_chol(chol, right.T).T
        xs[t] = xf[t] + j_gain @ (xs[t + 1] - xp[t + 1])
        ps[t] = _symmetrize(
            pf[t] + j_gain @ (ps[t + 1] - pp[t + 1]) @ j_gain.T
        )
        ps[t] = _nearest_psd(ps[t], floor=numerical_jitter)

    return xs, ps


# -----------------------------------------------------------------------------
# 参数向量化与 MLE
# -----------------------------------------------------------------------------


def _pack_lower_cholesky(a: Array) -> Array:
    n = a.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1):
            value = float(a[i, j])
            out.append(math.log(max(value, 1e-12)) if i == j else value)
    return np.asarray(out, float)


def _unpack_lower_cholesky(theta: Array, n: int) -> tuple[Array, int]:
    l = np.zeros((n, n), dtype=float)
    pos = 0
    for i in range(n):
        for j in range(i + 1):
            raw = float(theta[pos])
            l[i, j] = math.exp(raw) if i == j else raw
            pos += 1
    return l, pos


def _inverse_logit_bounded(value: float, low: float, high: float) -> float:
    eps = 1e-10
    ratio = (value - low) / (high - low)
    ratio = float(np.clip(ratio, eps, 1.0 - eps))
    return math.log(ratio / (1.0 - ratio))


def _logit_bounded(raw: float, low: float, high: float) -> float:
    if raw >= 0:
        exp_neg = math.exp(-raw)
        sig = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(raw)
        sig = exp_pos / (1.0 + exp_pos)
    return low + (high - low) * sig


def pack_parameters(
    params: WXParams,
    lb_spec: LBSpec,
    model_spec: ModelSpec,
) -> Array:
    n = len(params.cP)
    m = params.R.shape[0]
    parts: list[Array] = [np.asarray(params.cP, float)]

    if model_spec.diagonal_dynamics:
        parts.append(np.arctanh(np.clip(np.diag(params.AP), -0.995, 0.995)))
    else:
        parts.append(np.asarray(params.AP, float).ravel())

    parts.append(np.asarray(params.cQ, float))
    if model_spec.diagonal_dynamics:
        parts.append(np.arctanh(np.clip(np.diag(params.AQ), -0.995, 0.995)))
    else:
        parts.append(np.asarray(params.AQ, float).ravel())

    parts.append(_pack_lower_cholesky(params.Sigma))

    if model_spec.estimate_delta:
        parts.append(np.array([params.delta0], dtype=float))
        parts.append(np.asarray(params.delta1, float))

    if model_spec.diagonal_measurement:
        parts.append(np.log(np.maximum(np.diag(params.R), 1e-12)))
    else:
        chol_r = np.linalg.cholesky(_nearest_psd(params.R, floor=1e-12))
        parts.append(_pack_lower_cholesky(chol_r))

    if lb_spec.mode.strip().lower() == "estimated":
        low, high = lb_spec.bounds
        parts.append(
            np.asarray(
                [_inverse_logit_bounded(v, low, high) for v in params.lb_params],
                float,
            )
        )

    if (
        model_spec.curve_control_mode.lower() == "observedcontrols"
        and model_spec.estimate_controls
        and params.Gamma is not None
    ):
        parts.append(params.Gamma.ravel())

    return np.concatenate(parts)


def unpack_parameters(
    theta: Array,
    template: WXParams,
    lb_spec: LBSpec,
    model_spec: ModelSpec,
) -> WXParams:
    th = np.asarray(theta, float)
    n = len(template.cP)
    m = template.R.shape[0]
    pos = 0

    c_p = th[pos : pos + n]
    pos += n

    if model_spec.diagonal_dynamics:
        ap = np.diag(np.tanh(th[pos : pos + n]))
        pos += n
    else:
        raw = th[pos : pos + n * n].reshape(n, n)
        pos += n * n
        rho = _spectral_radius(raw)
        ap = raw / max(1.001, rho + 1e-3)

    c_q = th[pos : pos + n]
    pos += n

    if model_spec.diagonal_dynamics:
        aq = np.diag(np.tanh(th[pos : pos + n]))
        pos += n
    else:
        raw = th[pos : pos + n * n].reshape(n, n)
        pos += n * n
        rho = _spectral_radius(raw)
        aq = raw / max(1.001, rho + 1e-3)

    sigma, used = _unpack_lower_cholesky(th[pos:], n)
    pos += used

    if model_spec.estimate_delta:
        delta0 = float(th[pos])
        pos += 1
        delta1 = th[pos : pos + n]
        pos += n
    else:
        delta0 = template.delta0
        delta1 = template.delta1.copy()

    if model_spec.diagonal_measurement:
        r = np.diag(np.exp(th[pos : pos + m]))
        pos += m
    else:
        chol_r, used = _unpack_lower_cholesky(th[pos:], m)
        pos += used
        r = chol_r @ chol_r.T

    lb_params = template.lb_params.copy()
    if lb_spec.mode.strip().lower() == "estimated":
        count = len(template.lb_params)
        low, high = lb_spec.bounds
        lb_params = np.asarray(
            [_logit_bounded(v, low, high) for v in th[pos : pos + count]],
            float,
        )
        pos += count

    gamma = None if template.Gamma is None else template.Gamma.copy()
    if (
        model_spec.curve_control_mode.lower() == "observedcontrols"
        and model_spec.estimate_controls
        and gamma is not None
    ):
        count = gamma.size
        gamma = th[pos : pos + count].reshape(gamma.shape)
        pos += count

    if pos != len(th):
        raise ValueError(f"Parameter vector length mismatch: consumed {pos}, got {len(th)}")

    return WXParams(
        cP=c_p,
        AP=ap,
        cQ=c_q,
        AQ=aq,
        Sigma=sigma,
        delta0=delta0,
        delta1=delta1,
        R=r,
        lb_params=lb_params,
        Gamma=gamma,
    )


def _lb_prior_penalty(params: WXParams, lb_spec: LBSpec) -> float:
    if lb_spec.mode.strip().lower() != "estimated":
        return 0.0
    if lb_spec.prior_mean is None or lb_spec.prior_sd is None:
        return 0.0

    count = len(params.lb_params)
    mean = np.asarray(lb_spec.prior_mean, float)
    sd = np.asarray(lb_spec.prior_sd, float)
    mean = np.full(count, float(mean)) if mean.ndim == 0 else np.broadcast_to(mean, (count,))
    sd = np.full(count, float(sd)) if sd.ndim == 0 else np.broadcast_to(sd, (count,))
    if np.any(sd <= 0):
        raise ValueError("LB prior_sd must be positive")
    return float(0.5 * np.sum(((params.lb_params - mean) / sd) ** 2))


def fit_mle_multistart(
    forward_obs: Array,
    interpolation_flags: Array,
    forward_starts: Array,
    forward_tenor: float,
    initial_params: WXParams,
    x0: Array,
    p0: Array,
    lb_spec: LBSpec,
    model_spec: ModelSpec,
    controls: Array | None,
) -> tuple[WXParams, dict[str, Any]]:
    theta0 = pack_parameters(initial_params, lb_spec, model_spec)
    rng = np.random.default_rng(model_spec.random_seed)

    def objective(theta: Array) -> float:
        try:
            params = unpack_parameters(theta, initial_params, lb_spec, model_spec)
            out = ekf_wuxia(
                forward_obs=forward_obs,
                interpolation_flags=interpolation_flags,
                forward_starts=forward_starts,
                forward_tenor=forward_tenor,
                params=params,
                x0=x0,
                p0=p0,
                lb_spec=lb_spec,
                model_spec=model_spec,
                controls=controls,
            )
            value = -float(out["loglik"])
            value += _lb_prior_penalty(params, lb_spec)
            if (
                model_spec.estimate_controls
                and params.Gamma is not None
                and params.Gamma.size > 0
            ):
                value += model_spec.control_ridge * float(np.sum(params.Gamma**2))
            if not np.isfinite(value):
                return 1e100
            return value
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
            return 1e100

    candidates: list[dict[str, Any]] = []
    starts = max(1, int(model_spec.num_starts))
    for s in range(starts):
        if s == 0:
            start = theta0.copy()
        else:
            scale = 0.03 + 0.02 * s
            start = theta0 + rng.normal(0.0, scale, size=len(theta0))

        opt = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={
                "maxiter": int(model_spec.maxiter),
                "ftol": float(model_spec.optimizer_ftol),
                "maxls": 40,
            },
        )
        candidates.append(
            {
                "success": bool(opt.success),
                "fun": float(opt.fun),
                "message": str(opt.message),
                "nit": int(getattr(opt, "nit", -1)),
                "x": np.asarray(opt.x, float),
            }
        )

    finite = [c for c in candidates if np.isfinite(c["fun"])]
    if not finite:
        raise RuntimeError("All MLE starts failed")
    best = min(finite, key=lambda c: c["fun"])
    fitted = unpack_parameters(best["x"], initial_params, lb_spec, model_spec)
    diagnostics = {
        "best_objective": best["fun"],
        "best_success": best["success"],
        "best_message": best["message"],
        "best_iterations": best["nit"],
        "all_starts": [
            {k: v for k, v in c.items() if k != "x"}
            for c in candidates
        ],
    }
    return fitted, diagnostics


# -----------------------------------------------------------------------------
# 输出、拟合诊断与政策变量比较
# -----------------------------------------------------------------------------


def _forward_fit_from_state(
    states: Array,
    lb_path: Array,
    forward_starts: Array,
    forward_tenor: float,
    params: WXParams,
    model_spec: ModelSpec,
    controls: Array | None,
) -> Array:
    t_len = len(states)
    m_len = len(forward_starts)
    tenor_steps = steps_from_maturity(forward_tenor, model_spec.state_step)
    fit = np.full((t_len, m_len), np.nan)
    all_idx = np.arange(m_len)
    for t in range(t_len):
        fit[t], _ = observation_vector_and_jacobian(
            t=t,
            x=states[t],
            lb=float(lb_path[t]),
            indices=all_idx,
            forward_starts=forward_starts,
            tenor_steps=tenor_steps,
            params=params,
            model_spec=model_spec,
            controls=controls,
        )
    return fit


def compute_fit_diagnostics(
    forward_obs: Array,
    forward_fit: Array,
    forward_starts: Array,
    interpolation_flags: Array,
) -> pd.DataFrame:
    rows = []
    for j, maturity in enumerate(forward_starts):
        valid = np.isfinite(forward_obs[:, j]) & np.isfinite(forward_fit[:, j])
        if np.sum(valid) == 0:
            rmse = mae = max_abs = np.nan
        else:
            err = forward_obs[valid, j] - forward_fit[valid, j]
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            max_abs = float(np.max(np.abs(err)))
        rows.append(
            {
                "forward_start_year": float(maturity),
                "RMSE": rmse,
                "MAE": mae,
                "MaxAbsError": max_abs,
                "ObservationCount": int(np.sum(valid)),
                "InterpolationShare": float(
                    np.mean(interpolation_flags[valid, j]) if np.sum(valid) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _lead_lag_corr(x: Array, y: Array, lag: int) -> float:
    if lag > 0:
        xx, yy = x[:-lag], y[lag:]
    elif lag < 0:
        xx, yy = x[-lag:], y[:lag]
    else:
        xx, yy = x, y
    valid = np.isfinite(xx) & np.isfinite(yy)
    if np.sum(valid) < 3:
        return np.nan
    return float(np.corrcoef(xx[valid], yy[valid])[0, 1])


def compute_compare_diagnostics(
    shadow: Array,
    nominal: Array,
    compare_series: Mapping[str, Sequence[float]] | None,
    max_lag: int = 6,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not compare_series:
        return out

    t_len = len(shadow)
    for name, values in compare_series.items():
        series = np.asarray(values, float)[:t_len]
        n = min(t_len, len(series))
        s = shadow[:n]
        r = nominal[:n]
        z = series[:n]
        valid = np.isfinite(s) & np.isfinite(z)
        if np.sum(valid) < 3:
            continue

        item: dict[str, Any] = {
            "CorrelationShadow": float(np.corrcoef(s[valid], z[valid])[0, 1]),
            "MeanDiffShadow": float(np.mean(s[valid] - z[valid])),
            "MeanDiffNominal": float(np.mean(r[valid] - z[valid])),
            "LeadLagCorrelation": {
                int(lag): _lead_lag_corr(s, z, lag)
                for lag in range(-max_lag, max_lag + 1)
            },
        }
        out[name] = item
    return out


def _assemble_result(
    forward_obs: Array,
    interpolation_flags: Array,
    zero_filled: Array,
    forward_starts: Array,
    forward_tenor: float,
    params: WXParams,
    filter_out: dict[str, Any],
    smooth: bool,
    lb_spec: LBSpec,
    model_spec: ModelSpec,
    controls: Array | None,
    compare_series: Mapping[str, Sequence[float]] | None,
    mle_diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if smooth:
        state, state_cov = rts_smoother_affine(
            filter_out,
            params.AP,
            model_spec.numerical_jitter,
        )
    else:
        state = np.asarray(filter_out["x_filt"], float)
        state_cov = np.asarray(filter_out["P_filt"], float)

    lb_path = np.asarray(filter_out["LBPath"], float)
    shadow = params.delta0 + state @ params.delta1
    nominal = np.maximum(shadow, lb_path)

    shadow_var = np.einsum("i,tij,j->t", params.delta1, state_cov, params.delta1)
    shadow_se = np.sqrt(np.maximum(shadow_var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_bind = np.where(
            shadow_se > 0,
            (lb_path - shadow) / shadow_se,
            np.where(shadow <= lb_path, np.inf, -np.inf),
        )
    bind_prob = ndtr(z_bind)

    alpha = 1.0 - model_spec.ci_level
    zcrit = float(ndtri(1.0 - alpha / 2.0))
    shadow_ci = np.column_stack(
        [shadow - zcrit * shadow_se, shadow + zcrit * shadow_se]
    )

    forward_fit = _forward_fit_from_state(
        states=state,
        lb_path=lb_path,
        forward_starts=forward_starts,
        forward_tenor=forward_tenor,
        params=params,
        model_spec=model_spec,
        controls=controls,
    )
    fit_diag = compute_fit_diagnostics(
        forward_obs,
        forward_fit,
        forward_starts,
        interpolation_flags,
    )
    compare_diag = compute_compare_diagnostics(
        shadow,
        nominal,
        compare_series,
    )

    return {
        "rShadow": shadow,
        "rNominal": nominal,
        "LBPath": lb_path,
        "BindProb": bind_prob,
        "ShadowSE": shadow_se,
        "ShadowCI": shadow_ci,
        "StatePath": state,
        "StateCovariance": state_cov,
        "FilteredState": filter_out["x_filt"],
        "FilteredStateCovariance": filter_out["P_filt"],
        "PredictedState": filter_out["x_pred"],
        "PredictedStateCovariance": filter_out["P_pred"],
        "ForwardObs": forward_obs,
        "ForwardFit": forward_fit,
        "ZeroYieldFilled": zero_filled,
        "InterpolationFlags": interpolation_flags,
        "LogLikelihood": float(filter_out["loglik"]),
        "EstimatedParameters": params.as_dict(),
        "FitDiag": fit_diag,
        "CompareDiag": compare_diag,
        "MLEDiagnostics": None if mle_diagnostics is None else dict(mle_diagnostics),
        "ModelLabel": "China-adapted Wu-Xia shadow-rate term-structure model",
    }


# -----------------------------------------------------------------------------
# 主入口：FixParam / MLE / RollingMLE
# -----------------------------------------------------------------------------


def _validate_inputs(
    zero_yields: Array,
    curve_maturities: Array,
    forward_starts: Array,
    model_spec: ModelSpec,
) -> None:
    if zero_yields.shape[1] != len(curve_maturities):
        raise ValueError("zero_yields maturity dimension mismatch")
    if model_spec.state_dim <= 0:
        raise ValueError("state_dim must be positive")
    if model_spec.state_step <= 0:
        raise ValueError("state_step must be positive")
    if np.any(np.diff(curve_maturities) <= 0):
        raise ValueError("curve_maturities must be strictly increasing")
    if np.any(forward_starts < 0):
        raise ValueError("forward_starts must be nonnegative")


def _coerce_init_state(
    init_state: Mapping[str, Any] | None,
    fallback_x0: Array,
    fallback_p0: Array,
    n: int,
) -> tuple[Array, Array]:
    raw = {} if init_state is None else dict(init_state)
    x0 = np.asarray(raw.get("x0", raw.get("a0", fallback_x0)), float)
    p0 = np.asarray(raw.get("P0", fallback_p0), float)
    if x0.shape != (n,):
        raise ValueError(f"x0 must have shape {(n,)}")
    _check_square(p0, n, "P0")
    return x0, _nearest_psd(p0, floor=1e-8)


def estimate_shadow_rate(
    zero_yields: Array,
    curve_maturities: Sequence[float],
    forward_starts: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
    forward_tenor: float = 1.0 / 12.0,
    lb_spec: LBSpec | Mapping[str, Any] | None = None,
    model_spec: ModelSpec | Mapping[str, Any] | None = None,
    est_mode: str = "FixParam",
    param0: Mapping[str, Any] | WXParams | None = None,
    init_state: Mapping[str, Any] | None = None,
    missing_rule: str = "CurveInterpolate",
    smooth: bool = True,
    control_series: Mapping[str, Sequence[float]] | None = None,
    compare_series: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """估计中国国债期限结构隐含的 Wu–Xia 影子短端利率。

    Notes
    -----
    1. zero_yields 必须为连续复利、年化小数口径；
    2. 若原始数据是附息国债到期收益率，应先在本函数外构建零息曲线；
    3. est_mode 可取 FixParam、MLE 或 RollingMLE；
    4. RollingMLE 返回实时滚动估计路径，不使用全样本平滑信息。
    """

    y = _as_2d_float(zero_yields, "zero_yields")
    curve_mats = _as_1d_float(curve_maturities, "curve_maturities")
    starts = _as_1d_float(forward_starts, "forward_starts")

    if lb_spec is None:
        lb_cfg = LBSpec()
    elif isinstance(lb_spec, LBSpec):
        lb_cfg = lb_spec
    else:
        lb_cfg = LBSpec(**dict(lb_spec))

    if model_spec is None:
        model_cfg = ModelSpec()
    elif isinstance(model_spec, ModelSpec):
        model_cfg = model_spec
    else:
        model_cfg = ModelSpec(**dict(model_spec))

    _validate_inputs(y, curve_mats, starts, model_cfg)

    forward_obs, interpolation_flags, zero_filled = build_forward_observations(
        zero_yields=y,
        curve_maturities=curve_mats,
        forward_starts=starts,
        forward_tenor=forward_tenor,
        missing_rule=missing_rule,
    )

    controls = build_control_matrix(control_series, model_cfg, len(y))
    initial_params, fallback_x0, fallback_p0 = initialize_parameters(
        forward_obs=forward_obs,
        forward_starts=starts,
        model_spec=model_cfg,
        lb_spec=lb_cfg,
        controls=controls,
        param0=param0,
    )
    x0, p0 = _coerce_init_state(
        init_state,
        fallback_x0,
        fallback_p0,
        model_cfg.state_dim,
    )

    mode = est_mode.strip().upper()
    if mode == "ROLLINGMLE":
        return estimate_shadow_rate_rolling(
            zero_yields=y,
            curve_maturities=curve_mats,
            forward_starts=starts,
            forward_tenor=forward_tenor,
            lb_spec=lb_cfg,
            model_spec=model_cfg,
            param0=initial_params,
            init_state={"x0": x0, "P0": p0},
            missing_rule=missing_rule,
            control_series=control_series,
            compare_series=compare_series,
        )

    mle_diag = None
    params = initial_params
    if mode == "MLE":
        params, mle_diag = fit_mle_multistart(
            forward_obs=forward_obs,
            interpolation_flags=interpolation_flags,
            forward_starts=starts,
            forward_tenor=forward_tenor,
            initial_params=initial_params,
            x0=x0,
            p0=p0,
            lb_spec=lb_cfg,
            model_spec=model_cfg,
            controls=controls,
        )
    elif mode != "FIXPARAM":
        raise ValueError("est_mode must be FixParam, MLE, or RollingMLE")

    filter_out = ekf_wuxia(
        forward_obs=forward_obs,
        interpolation_flags=interpolation_flags,
        forward_starts=starts,
        forward_tenor=forward_tenor,
        params=params,
        x0=x0,
        p0=p0,
        lb_spec=lb_cfg,
        model_spec=model_cfg,
        controls=controls,
    )

    return _assemble_result(
        forward_obs=forward_obs,
        interpolation_flags=interpolation_flags,
        zero_filled=zero_filled,
        forward_starts=starts,
        forward_tenor=forward_tenor,
        params=params,
        filter_out=filter_out,
        smooth=smooth,
        lb_spec=lb_cfg,
        model_spec=model_cfg,
        controls=controls,
        compare_series=compare_series,
        mle_diagnostics=mle_diag,
    )


def _slice_lb_spec(lb_spec: LBSpec, start: int, end: int) -> LBSpec:
    if lb_spec.regime_series is None:
        return lb_spec
    sliced = np.asarray(lb_spec.regime_series)[start:end].tolist()
    return replace(lb_spec, regime_series=sliced)


def _slice_mapping(
    values: Mapping[str, Sequence[float]] | None,
    start: int,
    end: int,
) -> dict[str, Array] | None:
    if values is None:
        return None
    return {name: np.asarray(series, float)[start:end] for name, series in values.items()}


def estimate_shadow_rate_rolling(
    zero_yields: Array,
    curve_maturities: Sequence[float],
    forward_starts: Sequence[float],
    forward_tenor: float,
    lb_spec: LBSpec,
    model_spec: ModelSpec,
    param0: WXParams | Mapping[str, Any] | None,
    init_state: Mapping[str, Any] | None,
    missing_rule: str,
    control_series: Mapping[str, Sequence[float]] | None,
    compare_series: Mapping[str, Sequence[float]] | None,
) -> dict[str, Any]:
    """滚动 MLE：每个时点只使用窗口内及此前可获得的信息。"""

    y = np.asarray(zero_yields, float)
    t_len = len(y)
    window = model_spec.rolling_window
    if window is None:
        window = min(max(120, 5 * model_spec.state_dim * len(forward_starts)), t_len)
    if window < 20 or window > t_len:
        raise ValueError("rolling_window must be between 20 and T")
    step = max(1, int(model_spec.rolling_step))

    shadow = np.full(t_len, np.nan)
    nominal = np.full(t_len, np.nan)
    lb_path = np.full(t_len, np.nan)
    bind_prob = np.full(t_len, np.nan)
    shadow_se = np.full(t_len, np.nan)
    shadow_ci = np.full((t_len, 2), np.nan)
    loglik = np.full(t_len, np.nan)
    parameter_history: list[dict[str, Any]] = []

    warm_params = param0
    last_result: dict[str, Any] | None = None

    for end in range(window, t_len + 1, step):
        start = end - window
        local_lb = _slice_lb_spec(lb_spec, start, end)
        local_controls = _slice_mapping(control_series, start, end)
        local_compare = _slice_mapping(compare_series, start, end)
        local_model = replace(model_spec, rolling_window=None)

        result = estimate_shadow_rate(
            zero_yields=y[start:end],
            curve_maturities=curve_maturities,
            forward_starts=forward_starts,
            forward_tenor=forward_tenor,
            lb_spec=local_lb,
            model_spec=local_model,
            est_mode="MLE",
            param0=warm_params,
            init_state=init_state,
            missing_rule=missing_rule,
            smooth=False,
            control_series=local_controls,
            compare_series=local_compare,
        )

        idx = end - 1
        shadow[idx] = result["rShadow"][-1]
        nominal[idx] = result["rNominal"][-1]
        lb_path[idx] = result["LBPath"][-1]
        bind_prob[idx] = result["BindProb"][-1]
        shadow_se[idx] = result["ShadowSE"][-1]
        shadow_ci[idx] = result["ShadowCI"][-1]
        loglik[idx] = result["LogLikelihood"]
        parameter_history.append(
            {
                "window_start": start,
                "window_end": end,
                "parameters": result["EstimatedParameters"],
                "mle": result["MLEDiagnostics"],
            }
        )
        warm_params = WXParams(
            cP=np.asarray(result["EstimatedParameters"]["cP"], float),
            AP=np.asarray(result["EstimatedParameters"]["AP"], float),
            cQ=np.asarray(result["EstimatedParameters"]["cQ"], float),
            AQ=np.asarray(result["EstimatedParameters"]["AQ"], float),
            Sigma=np.asarray(result["EstimatedParameters"]["Sigma"], float),
            delta0=float(result["EstimatedParameters"]["Delta0"]),
            delta1=np.asarray(result["EstimatedParameters"]["Delta1"], float),
            R=np.asarray(result["EstimatedParameters"]["R"], float),
            lb_params=np.atleast_1d(
                np.asarray(result["EstimatedParameters"]["LBParam"], float)
            ),
            Gamma=(
                None
                if result["EstimatedParameters"]["Gamma"] is None
                else np.asarray(result["EstimatedParameters"]["Gamma"], float)
            ),
        )
        last_result = result

    if last_result is None:
        raise RuntimeError("RollingMLE produced no estimation window")

    # rolling_step > 1 时，保持未估计时点为 NaN，避免伪造实时信号。
    compare_diag = compute_compare_diagnostics(shadow, nominal, compare_series)
    return {
        "rShadow": shadow,
        "rNominal": nominal,
        "LBPath": lb_path,
        "BindProb": bind_prob,
        "ShadowSE": shadow_se,
        "ShadowCI": shadow_ci,
        "LogLikelihoodPath": loglik,
        "ParameterHistory": parameter_history,
        "CompareDiag": compare_diag,
        "ModelLabel": "Rolling real-time China-adapted Wu-Xia model",
        "RollingWindow": window,
        "RollingStep": step,
    }


# 附录编号统一入口
run_a3 = estimate_shadow_rate


if __name__ == "__main__":
    # 轻量级合成数据冒烟示例。正式研究应输入真实中国国债零息曲线。
    rng = np.random.default_rng(7)
    t_demo = 80
    curve_mats_demo = np.array(
        [0.0, 1 / 12, 0.25, 1 / 3, 0.5, 7 / 12, 1.0, 13 / 12, 2.0,
         25 / 12, 3.0, 37 / 12, 5.0, 61 / 12, 7.0, 85 / 12, 10.0,
         121 / 12]
    )
    base = 0.018 + 0.002 * np.sin(np.linspace(0, 6, t_demo))
    zero_demo = np.empty((t_demo, len(curve_mats_demo)))
    for t in range(t_demo):
        slope = 0.004 * np.exp(-curve_mats_demo / 4.0)
        zero_demo[t] = base[t] + slope + rng.normal(0.0, 0.00015, len(curve_mats_demo))

    result_demo = estimate_shadow_rate(
        zero_yields=zero_demo,
        curve_maturities=curve_mats_demo,
        est_mode="FixParam",
        lb_spec=LBSpec(mode="Fixed", value=0.0),
        model_spec=ModelSpec(num_starts=1, maxiter=20),
        smooth=True,
    )
    print(result_demo["FitDiag"].head())
    print("Shadow-rate tail:", result_demo["rShadow"][-3:])
