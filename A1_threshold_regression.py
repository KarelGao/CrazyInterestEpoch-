# -*- coding: utf-8 -*-
from __future__ import annotations
import math, copy, warnings, json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, lsq_linear
from scipy.special import logsumexp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
try:
    import statsmodels.api as sm
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False
try:
    import cvxpy as cp
    CVXPY_OK = True
except Exception:
    CVXPY_OK = False
from common import (BUCKETS, BUCKET_CN, _as_float_array, _finite_rows, _jsonable, _write_json, ols_rss, project_simplex, numerical_gradient, within_demean)
from config import ThresholdConfig, AllocationConfig, StageConfig, SystemConfig

"""A1 门槛回归模型及显著性检验算法

与附录算法编号一一对应。主要入口：hansen_threshold_regression。
"""

def hansen_threshold_regression(
    y: Sequence[float],
    X: Sequence[Sequence[float]],
    q: Sequence[float],
    config: ThresholdConfig | None = None,
    id_t: Sequence[Any] | None = None,
    time_t: Sequence[Any] | None = None,
) -> dict[str, Any]:
    cfg = config or ThresholdConfig()
    rng = np.random.default_rng(cfg.seed)
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    q = np.asarray(q, float).reshape(-1)
    if not (len(y) == len(X) == len(q)):
        raise ValueError("y, X, q length mismatch")

    mask = _finite_rows(y, X, q)
    y0, X0, q0 = y[mask], X[mask], q[mask]
    ids = np.asarray(id_t)[mask] if id_t is not None else None
    times = np.asarray(time_t)[mask] if time_t is not None else None

    if cfg.fe_flag not in {"none", "time", "entity", "twoway"}:
        raise ValueError("fe_flag must be none/time/entity/twoway")
    yfe, Xfe = y0.copy(), X0.copy()
    if cfg.fe_flag in {"entity", "twoway"}:
        if ids is None:
            raise ValueError("id_t required for entity/twoway FE")
        yfe, Xfe = within_demean(yfe, ids), within_demean(Xfe, ids)
    if cfg.fe_flag in {"time", "twoway"}:
        if times is None:
            raise ValueError("time_t required for time/twoway FE")
        yfe, Xfe = within_demean(yfe, times), within_demean(Xfe, times)

    T0, k = len(yfe), Xfe.shape[1]
    if T0 <= 3 * k + 8:
        raise ValueError(f"too few effective observations for threshold test: T={T0}, k={k}")

    qlow, qhigh = np.quantile(q0, [cfg.trimming, 1 - cfg.trimming])
    if cfg.grid_type == "uniqueQ":
        grid = np.unique(q0[(q0 >= qlow) & (q0 <= qhigh)])
    else:
        probs = np.linspace(cfg.trimming, 1 - cfg.trimming, cfg.grid_points)
        grid = np.unique(np.quantile(q0, probs))
    min_seg = max(k + 2, int(np.ceil(cfg.trimming * T0)))
    grid = np.array([g for g in grid if np.sum(q0 <= g) >= min_seg and np.sum(q0 > g) >= min_seg], float)
    if len(grid) < 3:
        raise ValueError("threshold grid too small after trimming")

    def build_z(gamma: float) -> np.ndarray:
        i1 = (q0 <= gamma).astype(float)[:, None]
        i2 = 1.0 - i1
        return np.hstack([Xfe * i1, Xfe * i2])

    rss_list = []
    beta_list = []
    for gamma in grid:
        rss, beta, _ = ols_rss(build_z(gamma), yfe)
        rss_list.append(rss); beta_list.append(beta)
    rss_arr = np.asarray(rss_list)
    i_hat = int(np.argmin(rss_arr))
    gamma_hat = float(grid[i_hat])
    rss_hat = float(rss_arr[i_hat])
    sigma2_hat = rss_hat / max(T0 - 2 * k, 1)

    rss0, beta0, uhat = ols_rss(Xfe, yfe)
    den = np.maximum(rss_arr / max(T0 - 2 * k, 1), 1e-15)
    f_grid = ((rss0 - rss_arr) / k) / den
    supf_obs = float(np.nanmax(f_grid))

    def draw_resid(u: np.ndarray) -> np.ndarray:
        uc = u - np.mean(u)
        if cfg.boot_type == "residual":
            return rng.choice(uc, size=len(uc), replace=True)
        return uc * rng.choice([-1.0, 1.0], size=len(uc), replace=True)

    supf_boot = np.empty(cfg.bootstrap_B)
    suplr_boot = np.empty(cfg.bootstrap_B)
    for b in range(cfg.bootstrap_B):
        ustar = draw_resid(uhat)
        ystar = Xfe @ beta0 + ustar
        rss0s, _, _ = ols_rss(Xfe, ystar)
        rsss = np.array([ols_rss(build_z(g), ystar)[0] for g in grid])
        den_s = np.maximum(rsss / max(T0 - 2 * k, 1), 1e-15)
        supf_boot[b] = np.max(((rss0s - rsss) / k) / den_s)
        rmin = np.min(rsss)
        sig2s = max(rmin / max(T0 - 2 * k, 1), 1e-15)
        suplr_boot[b] = np.max((rsss - rmin) / sig2s)

    crit_supf = {str(a): float(np.quantile(supf_boot, 1 - a)) for a in cfg.alpha_list}
    pval = float((np.sum(supf_boot >= supf_obs) + 1) / (len(supf_boot) + 1))
    decision1 = "单门槛显著" if pval < 0.05 else "单门槛不显著"

    lr_grid = (rss_arr - rss_hat) / max(sigma2_hat, 1e-15)
    crit_lr = {str(a): float(np.quantile(suplr_boot, 1 - a)) for a in cfg.alpha_list}
    ci_gamma = {}
    for a in cfg.alpha_list:
        ok = grid[lr_grid <= crit_lr[str(a)]]
        ci_gamma[str(a)] = [float(np.min(ok)), float(np.max(ok))] if len(ok) else [gamma_hat, gamma_hat]

    decision2 = "未执行双门槛检验"
    f2obs = np.nan; pval2 = np.nan; gamma2_hat = np.nan
    if cfg.double_threshold and decision1 == "单门槛显著" and T0 > 4 * k + 8:
        def build_z2(g1: float, g2: float) -> np.ndarray:
            lo, hi = sorted([g1, g2])
            i1 = (q0 <= lo).astype(float)[:, None]
            i2 = ((q0 > lo) & (q0 <= hi)).astype(float)[:, None]
            i3 = (q0 > hi).astype(float)[:, None]
            return np.hstack([Xfe * i1, Xfe * i2, Xfe * i3])

        valid_g2 = []
        rss2 = []
        for g2 in grid:
            if abs(g2 - gamma_hat) < 1e-12:
                continue
            lo, hi = sorted([gamma_hat, g2])
            counts = [np.sum(q0 <= lo), np.sum((q0 > lo) & (q0 <= hi)), np.sum(q0 > hi)]
            if min(counts) < max(k + 2, int(np.ceil(cfg.trimming * T0 / 2))):
                continue
            valid_g2.append(g2)
            rss2.append(ols_rss(build_z2(gamma_hat, g2), yfe)[0])
        if valid_g2:
            rss2 = np.asarray(rss2)
            j = int(np.argmin(rss2))
            gamma2_hat = float(valid_g2[j])
            rss2hat = float(rss2[j])
            f2obs = float(((rss_hat - rss2hat) / k) / max(rss2hat / max(T0 - 3*k, 1), 1e-15))

            z1 = build_z(gamma_hat)
            rss1_obs, b1, resid1 = ols_rss(z1, yfe)
            f2boot = []
            for _ in range(cfg.bootstrap_B):
                ys = z1 @ b1 + draw_resid(resid1)
                rss1s = ols_rss(z1, ys)[0]
                r2s = min(ols_rss(build_z2(gamma_hat, g2), ys)[0] for g2 in valid_g2)
                f2s = ((rss1s - r2s) / k) / max(r2s / max(T0 - 3*k, 1), 1e-15)
                f2boot.append(f2s)
            f2boot = np.asarray(f2boot)
            pval2 = float((np.sum(f2boot >= f2obs) + 1) / (len(f2boot) + 1))
            decision2 = "双门槛显著" if pval2 < 0.05 else "双门槛不显著（顺序检验下停止）"

    grid_df = pd.DataFrame({"gamma": grid, "RSS": rss_arr, "F": f_grid, "LR": lr_grid})
    return {
        "gammaHat": gamma_hat,
        "gamma2Hat": gamma2_hat,
        "CIgamma": ci_gamma,
        "SupFobs": supf_obs,
        "pValue": pval,
        "CritSupF": crit_supf,
        "CritLR": crit_lr,
        "Decision1": decision1,
        "F2obs": f2obs,
        "pValue2": pval2,
        "Decision2": decision2,
        "rssGrid": grid_df,
        "betaHat": beta_list[i_hat],
        "T_eff": T0,
        "k": k,
    }

# 附录编号统一入口
run_a1 = hansen_threshold_regression
