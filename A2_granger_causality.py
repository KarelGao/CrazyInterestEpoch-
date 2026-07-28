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

"""A2 Granger 因果关系检验算法

与附录算法编号一一对应。主要入口：granger_stage_comparison。
"""

def apply_transform(s: Sequence[float], rule: str | Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    a = np.asarray(s, float)
    if callable(rule):
        return np.asarray(rule(a), float)
    if rule == "level":
        return a.copy()
    if rule == "diff":
        return np.diff(a)
    if rule == "logdiff":
        if np.any(a <= 0):
            raise ValueError("logdiff requires positive series")
        return np.diff(np.log(a))
    raise ValueError(f"unknown transform: {rule}")


def lag_matrix(s: np.ndarray, p: int) -> np.ndarray:
    s = np.asarray(s, float).reshape(-1)
    if p <= 0:
        return np.empty((len(s), 0))
    if len(s) <= p:
        return np.empty((0, p))
    return np.column_stack([s[p-j:len(s)-j] for j in range(1, p+1)])


def _align_xy(x: np.ndarray, y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    return y[p:], lag_matrix(y, p), lag_matrix(x, p)


def _info_criterion(rss: float, n: int, k: int, ic_type: str) -> float:
    rss = max(float(rss), 1e-15)
    if ic_type.upper() == "AIC":
        return np.log(rss/n) + 2*k/n
    return np.log(rss/n) + np.log(n)*k/n


def choose_granger_lag(x: np.ndarray, y: np.ndarray, pmax: int = 8, ic_type: str = "BIC") -> int:
    n = min(len(x), len(y))
    max_admissible = max(1, min(pmax, (n - 4)//3))
    scores = []
    for p in range(1, max_admissible + 1):
        yd, yl, xl = _align_xy(x, y, p)
        if len(yd) <= 2*p + 2:
            continue
        Z = np.column_stack([np.ones(len(yd)), yl, xl])
        rss = ols_rss(Z, yd)[0]
        scores.append((p, _info_criterion(rss, len(yd), Z.shape[1], ic_type)))
    return min(scores, key=lambda z: z[1])[0] if scores else 1


def granger_one_direction(x: np.ndarray, y: np.ndarray, p: int) -> dict[str, float]:
    yd, yl, xl = _align_xy(x, y, p)
    n = len(yd)
    ZU = np.column_stack([np.ones(n), yl, xl])
    ZR = np.column_stack([np.ones(n), yl])
    rssu = ols_rss(ZU, yd)[0]
    rssr = ols_rss(ZR, yd)[0]
    df1 = p
    df2 = n - ZU.shape[1]
    if df2 <= 0 or rssu <= 0:
        return {"p": p, "F": np.nan, "pValue": np.nan}
    fstat = max(0.0, ((rssr - rssu)/df1) / (rssu/df2))
    pval = float(stats.f.sf(fstat, df1, df2))
    return {"p": p, "F": float(fstat), "pValue": pval}


def significance_mark(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def granger_stage_comparison(
    x_t: Sequence[float], y_t: Sequence[float], stage_t: Sequence[str],
    p_max: int = 8, ic_type: str = "BIC", transform_rule: str = "level",
    allow_stage_lag: bool = False,
) -> pd.DataFrame:
    x0 = np.asarray(x_t, float); y0 = np.asarray(y_t, float); stage0 = np.asarray(stage_t, object)
    groups = ["Full", "Enter", "Visible", "Deep"]
    rows = []

    # Transform full sample once for the globally aligned version.
    xg, yg = apply_transform(x0, transform_rule), apply_transform(y0, transform_rule)
    drop = len(x0) - len(xg)
    sg = stage0[drop:drop+min(len(xg), len(yg))]
    xg, yg = xg[:len(sg)], yg[:len(sg)]

    for g in groups:
        raw_idx = np.arange(len(x0)) if g == "Full" else np.where(stage0 == g)[0]
        if len(raw_idx) < 8:
            rows.extend([
                {"Group": g, "Direction": "x -> y", "LagOrder": np.nan, "F": np.nan, "pValue": np.nan, "Sig": "", "N": len(raw_idx)},
                {"Group": g, "Direction": "y -> x", "LagOrder": np.nan, "F": np.nan, "pValue": np.nan, "Sig": "", "N": len(raw_idx)},
            ])
            continue
        xt = apply_transform(x0[raw_idx], transform_rule)
        yt = apply_transform(y0[raw_idx], transform_rule)
        n = min(len(xt), len(yt)); xt, yt = xt[:n], yt[:n]
        p = choose_granger_lag(xt, yt, p_max, ic_type)

        if allow_stage_lag or g == "Full":
            out_xy = granger_one_direction(xt, yt, p)
            out_yx = granger_one_direction(yt, xt, p)
            for direction, out in [("x -> y", out_xy), ("y -> x", out_yx)]:
                rows.append({"Group": g, "Direction": direction, "LagOrder": out["p"], "F": out["F"],
                             "pValue": out["pValue"], "Sig": significance_mark(out["pValue"]), "N": n})
            continue

        # Global lag construction, then group slice at dependent-variable date.
        yd, yl, xl = _align_xy(xg, yg, p)
        stage_dep = sg[p:]
        idx = np.arange(len(yd)) if g == "Full" else np.where(stage_dep == g)[0]
        if len(idx) <= 2*p + 2:
            for direction in ["x -> y", "y -> x"]:
                rows.append({"Group": g, "Direction": direction, "LagOrder": p, "F": np.nan, "pValue": np.nan, "Sig": "", "N": len(idx)})
            continue
        # x -> y
        ydg, ylg, xlg = yd[idx], yl[idx], xl[idx]
        ZU = np.column_stack([np.ones(len(idx)), ylg, xlg]); ZR = np.column_stack([np.ones(len(idx)), ylg])
        rssu, rssr = ols_rss(ZU, ydg)[0], ols_rss(ZR, ydg)[0]
        df2 = len(idx) - ZU.shape[1]
        fxy = max(0, ((rssr-rssu)/p)/(rssu/df2)) if df2 > 0 and rssu > 0 else np.nan
        pxy = float(stats.f.sf(fxy, p, df2)) if np.isfinite(fxy) else np.nan
        rows.append({"Group": g, "Direction": "x -> y", "LagOrder": p, "F": fxy, "pValue": pxy, "Sig": significance_mark(pxy), "N": len(idx)})
        # y -> x on same aligned dates.
        xd = xg[p:][idx]
        ZU2 = np.column_stack([np.ones(len(idx)), xlg, ylg]); ZR2 = np.column_stack([np.ones(len(idx)), xlg])
        rssu2, rssr2 = ols_rss(ZU2, xd)[0], ols_rss(ZR2, xd)[0]
        df22 = len(idx) - ZU2.shape[1]
        fyx = max(0, ((rssr2-rssu2)/p)/(rssu2/df22)) if df22 > 0 and rssu2 > 0 else np.nan
        pyx = float(stats.f.sf(fyx, p, df22)) if np.isfinite(fyx) else np.nan
        rows.append({"Group": g, "Direction": "y -> x", "LagOrder": p, "F": fyx, "pValue": pyx, "Sig": significance_mark(pyx), "N": len(idx)})
    return pd.DataFrame(rows)

# 附录编号统一入口
run_a2 = granger_stage_comparison
