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

"""A3 Wu–Xia 影子利率估计算法

与附录算法编号一一对应。主要入口：estimate_shadow_rate。
"""

def nelson_siegel_loadings(maturities: np.ndarray, lam: float) -> np.ndarray:
    m = np.asarray(maturities, float)
    z = np.maximum(lam*m, 1e-12)
    l1 = (1 - np.exp(-z))/z
    l2 = l1 - np.exp(-z)
    return np.column_stack([np.ones(len(m)), l1, l2])


def _interpolate_curve(row: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    row = np.asarray(row, float).copy()
    ok = np.isfinite(row)
    if np.sum(ok) == 0:
        return np.zeros_like(row)
    if np.sum(ok) == 1:
        row[:] = row[ok][0]
        return row
    row[~ok] = np.interp(maturities[~ok], maturities[ok], row[ok])
    return row


def _kalman_filter_ns(
    Y: np.ndarray, maturities: np.ndarray, A: np.ndarray, Q: np.ndarray, R: np.ndarray, lam: float,
    a0: np.ndarray, P0: np.ndarray, missing_rule: str = "Interpolate",
) -> dict[str, Any]:
    Y = np.asarray(Y, float)
    T, M = Y.shape; N = len(a0)
    Hfull = nelson_siegel_loadings(maturities, lam)
    a = np.asarray(a0, float).copy(); P = np.asarray(P0, float).copy()
    a_pred = np.zeros((T, N)); a_filt = np.zeros((T, N))
    P_pred = np.zeros((T, N, N)); P_filt = np.zeros((T, N, N))
    ll = 0.0
    residual_full = np.full((T, M), np.nan)
    for t in range(T):
        ap = A @ a; Pp = A @ P @ A.T + Q
        yy = Y[t].copy()
        if missing_rule.lower() == "interpolate":
            yy = _interpolate_curve(yy, maturities); idx = np.arange(M)
        else:
            idx = np.where(np.isfinite(yy))[0]
            if len(idx) == 0:
                a, P = ap, Pp
                a_pred[t], a_filt[t] = ap, a
                P_pred[t], P_filt[t] = Pp, P
                continue
            yy = yy[idx]
        H = Hfull[idx]; Rt = R[np.ix_(idx, idx)]
        v = yy - H @ ap
        S = H @ Pp @ H.T + Rt
        S = (S + S.T)/2 + 1e-10*np.eye(len(idx))
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        K = Pp @ H.T @ Sinv
        a = ap + K @ v
        I = np.eye(N)
        P = (I - K@H)@Pp@(I-K@H).T + K@Rt@K.T  # Joseph form
        sign, logdet = np.linalg.slogdet(S)
        if sign > 0:
            ll += -0.5*(logdet + v@Sinv@v + len(v)*np.log(2*np.pi))
        a_pred[t], a_filt[t], P_pred[t], P_filt[t] = ap, a, Pp, P
        residual_full[t, idx] = v
    return {"a_pred": a_pred, "a_filt": a_filt, "P_pred": P_pred, "P_filt": P_filt,
            "loglik": float(ll), "residuals": residual_full, "H": Hfull}


def _rts_smoother(filter_out: dict[str, Any], A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    af = filter_out["a_filt"]; Pf = filter_out["P_filt"]
    ap = filter_out["a_pred"]; Pp = filter_out["P_pred"]
    T, N = af.shape
    ass = af.copy(); Pss = Pf.copy()
    for t in range(T-2, -1, -1):
        inv = np.linalg.pinv(Pp[t+1])
        J = Pf[t] @ A.T @ inv
        ass[t] = af[t] + J @ (ass[t+1] - ap[t+1])
        Pss[t] = Pf[t] + J @ (Pss[t+1] - Pp[t+1]) @ J.T
    return ass, Pss


def estimate_shadow_rate(
    Y: np.ndarray, maturities: Sequence[float], lb: float = 0.0, est_mode: str = "FixParam",
    param0: dict[str, Any] | None = None, init_state: dict[str, Any] | None = None,
    missing_rule: str = "Interpolate", smooth: bool = True, short_maturity: float = 0.25,
    compare_series: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    Y = np.asarray(Y, float); mats = np.asarray(maturities, float)
    T, M = Y.shape
    if M != len(mats): raise ValueError("Y maturity dimension mismatch")

    # Obtain a stable data-driven initial state and covariance from cross-sectional NS OLS.
    lam0 = float((param0 or {}).get("Lambda", 0.7))
    Yfill = np.vstack([_interpolate_curve(r, mats) for r in Y])
    H0 = nelson_siegel_loadings(mats, lam0)
    factors = np.array([np.linalg.lstsq(H0, row, rcond=None)[0] for row in Yfill])
    if T > 2:
        phi = []
        for j in range(3):
            x, z = factors[:-1, j], factors[1:, j]
            den = x@x
            phi.append(float(np.clip((x@z)/den if den>1e-12 else 0.95, -0.995, 0.995)))
        A0 = np.diag(phi)
        innov = factors[1:] - factors[:-1] @ A0.T
        Q0 = np.cov(innov.T) + 1e-5*np.eye(3)
    else:
        A0 = 0.95*np.eye(3); Q0 = 1e-3*np.eye(3)
    fitted = factors @ H0.T
    e = Yfill - fitted
    rdiag = np.maximum(np.nanvar(e, axis=0), 1e-4)
    R0 = np.diag(rdiag)

    p0 = param0 or {}
    A = np.asarray(p0.get("A", A0), float)
    Q = np.asarray(p0.get("Q", Q0), float)
    R = np.asarray(p0.get("R", R0), float)
    lam = lam0
    a0 = np.asarray((init_state or {}).get("a0", factors[0]), float)
    P0 = np.asarray((init_state or {}).get("P0", np.cov(factors.T) + 1e-3*np.eye(3)), float)

    if est_mode.upper() == "MLE":
        # Parsimonious constrained MLE: diagonal AR(A), diagonal Q/R, lambda.
        # This mirrors the appendix's reparameterisation idea while remaining robust for small samples.
        theta0 = np.r_[np.arctanh(np.clip(np.diag(A), -0.98, 0.98)),
                      np.log(np.maximum(np.diag(Q), 1e-8)),
                      np.log(np.maximum(np.diag(R), 1e-8)), np.log(max(lam, 1e-3))]
        def unpack(th):
            aa = np.diag(np.tanh(th[:3]))
            qq = np.diag(np.exp(th[3:6]))
            rr = np.diag(np.exp(th[6:6+M]))
            llambda = float(np.exp(th[-1]))
            return aa, qq, rr, llambda
        def nll(th):
            aa, qq, rr, llambda = unpack(th)
            if not (0.02 <= llambda <= 5.0): return 1e9
            return -_kalman_filter_ns(Y, mats, aa, qq, rr, llambda, a0, P0, missing_rule)["loglik"]
        opt = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": 300})
        if opt.success and np.isfinite(opt.fun):
            A, Q, R, lam = unpack(opt.x)

    filt = _kalman_filter_ns(Y, mats, A, Q, R, lam, a0, P0, missing_rule)
    state, Ps = _rts_smoother(filt, A) if smooth else (filt["a_filt"], filt["P_filt"])
    hs = nelson_siegel_loadings(np.array([short_maturity]), lam)[0]
    shadow = state @ hs
    nominal = np.maximum(shadow, lb)
    yhat = state @ nelson_siegel_loadings(mats, lam).T
    rmse = np.sqrt(np.nanmean((Y - yhat)**2, axis=0))
    fitdiag = pd.DataFrame({"maturity_year": mats, "RMSE": rmse})
    comp = {}
    if compare_series:
        for name, ser in compare_series.items():
            a = np.asarray(ser, float)[:T]
            idx = np.isfinite(a) & np.isfinite(shadow[:len(a)])
            if np.sum(idx) >= 3:
                comp[f"CorrShadow_{name}"] = float(np.corrcoef(shadow[:len(a)][idx], a[idx])[0,1])
                comp[f"MeanDiffShadow_{name}"] = float(np.mean(shadow[:len(a)][idx] - a[idx]))
    return {"rShadow": shadow, "rNominal": nominal, "StatePath": state, "StateCov": Ps,
            "LogLikelihood": filt["loglik"], "FitDiag": fitdiag, "CompareDiag": comp,
            "Param": {"A": A, "Q": Q, "R": R, "Lambda": lam}, "yHat": yhat}

# 附录编号统一入口
run_a3 = estimate_shadow_rate
