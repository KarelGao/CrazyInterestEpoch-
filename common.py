# -*- coding: utf-8 -*-
"""Shared constants and numerical utilities used by A1-A10 modules."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import pandas as pd

BUCKETS = ["credit", "bond", "ib", "trading", "cb_gov", "wm_fund"]
BUCKET_CN = {
    "credit": "信贷",
    "bond": "债券及其他债权",
    "ib": "同业及资金市场",
    "trading": "交易及衍生品",
    "cb_gov": "央行及政府类",
    "wm_fund": "理财/基金/协同",
}

def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    masks = []
    n = None
    for arr in arrays:
        a = np.asarray(arr)
        if n is None:
            n = len(a)
        if a.ndim == 1:
            masks.append(np.isfinite(a.astype(float)))
        else:
            masks.append(np.all(np.isfinite(a.astype(float)), axis=1))
    if not masks:
        return np.array([], dtype=bool)
    out = masks[0].copy()
    for m in masks[1:]:
        out &= m
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def ols_rss(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    rss = float(resid @ resid)
    return rss, beta, resid


def project_simplex(v: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """Euclidean projection on {x>=0, sum x=target_sum}."""
    v = np.asarray(v, float).reshape(-1)
    if target_sum <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - target_sum
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones_like(v) * target_sum / len(v)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def numerical_gradient(fun: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        h = eps * max(1.0, abs(x[i]))
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (fun(xp) - fun(xm)) / (2 * h)
    return g


def within_demean(v: np.ndarray, groups: Sequence[Any]) -> np.ndarray:
    v = np.asarray(v, float)
    g = np.asarray(groups)
    out = v.copy()
    for lab in pd.unique(g):
        idx = np.where(g == lab)[0]
        if v.ndim == 1:
            out[idx] = v[idx] - np.nanmean(v[idx])
        else:
            out[idx, :] = v[idx, :] - np.nanmean(v[idx, :], axis=0)
    return out

def build_stage_t_input(panel: pd.DataFrame) -> pd.Series:
    """Prepare the stage_t input required by A2/A4 without invoking A7.

    The appendix defines stage_t as an input to A2/A4.  Therefore the main
    runner prepares this exogenous three-stage label before A1-A10 are run.
    A7 later performs the formal external-rate-stage recognition and NIM
    calibration as its own independent algorithm.
    """
    lpr = panel["LPR1_pct"].astype(float)
    out = pd.Series(index=panel.index, dtype=object)
    out.loc[lpr >= 2.75] = "Enter"
    out.loc[(lpr < 2.75) & (lpr >= 2.25)] = "Visible"
    out.loc[lpr < 2.25] = "Deep"
    return out
