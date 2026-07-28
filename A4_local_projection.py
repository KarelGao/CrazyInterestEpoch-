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
from A2_granger_causality import apply_transform

"""A4 分阶段动态响应（Local Projection）算法

与附录算法编号一一对应。主要入口：local_projection_stage。
"""

def local_projection_stage(
    rate: Sequence[float], nim: Sequence[float], stage: Sequence[str], hmax: int = 8,
    p_ctrl: int = 2, x_ctrl: np.ndarray | None = None, transform_rule: str = "level",
    shock_scale: float = 0.01, ci_level: float = 0.95,
) -> pd.DataFrame:
    r0, n0, s0 = np.asarray(rate, float), np.asarray(nim, float), np.asarray(stage, object)
    r = apply_transform(r0, transform_rule); n = apply_transform(n0, transform_rule)
    drop = max(len(r0)-len(r), len(n0)-len(n))
    m = min(len(r), len(n)); r, n = r[:m], n[:m]; s = s0[drop:drop+m]
    if m < p_ctrl + hmax + 5:
        hmax = max(1, min(hmax, m - p_ctrl - 5))

    shock = np.diff(r)                   # shock at date t corresponds r[t]-r[t-1]
    dn = np.diff(n)
    # t indices on transformed level series. shock[j] is at t=j+1.
    rows = []
    zcrit = stats.norm.ppf((1+ci_level)/2)
    for st in ["Enter", "Visible", "Deep"]:
        for h in range(hmax):
            yy=[]; xx=[]
            for j in range(p_ctrl, len(shock)-h):
                t = j + 1
                if s[t] != st: continue
                # cumulative response in NIM from t-1 to t+h
                yresp = n[t+h] - n[t-1]
                controls = []
                for lag in range(1, p_ctrl+1):
                    controls.append(shock[j-lag])
                    controls.append(dn[j-lag])
                if x_ctrl is not None:
                    xc = np.asarray(x_ctrl, float)
                    if xc.ndim == 1: xc = xc[:,None]
                    original_t = drop + t
                    for lag in range(1, p_ctrl+1):
                        oi = original_t-lag
                        if 0 <= oi < len(xc): controls.extend(xc[oi].tolist())
                yy.append(yresp); xx.append([1.0, shock[j], *controls])
            if len(yy) < max(8, len(xx[0])+2 if xx else 8):
                rows.append({"Stage": st, "Horizon": h, "IRF": np.nan, "SE": np.nan, "CI_low": np.nan, "CI_high": np.nan, "N": len(yy)})
                continue
            yv, Xv = np.asarray(yy,float), np.asarray(xx,float)
            beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
            if STATSMODELS_OK:
                try:
                    fit = sm.OLS(yv, Xv).fit(cov_type="HAC", cov_kwds={"maxlags": max(1,h+1)})
                    se = float(fit.bse[1])
                except Exception:
                    resid = yv-Xv@beta; s2=(resid@resid)/max(len(yv)-Xv.shape[1],1)
                    se=float(np.sqrt(max(s2*np.linalg.pinv(Xv.T@Xv)[1,1],0)))
            else:
                resid = yv-Xv@beta; s2=(resid@resid)/max(len(yv)-Xv.shape[1],1)
                se=float(np.sqrt(max(s2*np.linalg.pinv(Xv.T@Xv)[1,1],0)))
            irf = float(beta[1]*shock_scale); se_scaled = se*abs(shock_scale)
            rows.append({"Stage":st,"Horizon":h,"IRF":irf,"SE":se_scaled,
                         "CI_low":irf-zcrit*se_scaled,"CI_high":irf+zcrit*se_scaled,"N":len(yv)})
    return pd.DataFrame(rows)

# 附录编号统一入口
run_a4 = local_projection_stage
