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

"""A7 外部利率阶段识别与 NIM 内部校准算法

与附录算法编号一一对应。主要入口：recognize_rate_stage。
"""

def _nearest_stage(v: float, centres: Sequence[float]) -> int:
    c=np.asarray(centres,float)
    return int(np.argmin(np.abs(c-float(v))))


def _persistence_filter(raw: Sequence[int], k: int) -> np.ndarray:
    raw=np.asarray(raw,int); out=raw.copy()
    if len(raw)==0:return out
    out[0]=raw[0]; candidate=raw[0]; count=0
    current=raw[0]
    for t in range(1,len(raw)):
        if raw[t]==current:
            candidate=current; count=0; out[t]=current
        else:
            if raw[t]==candidate: count+=1
            else: candidate=raw[t]; count=1
            if count>=max(k,1): current=candidate; count=0
            out[t]=current
    return out


def _nim_state_index(nim: float, thresholds: tuple[float,float,float]) -> int:
    a,b,c=thresholds
    if nim>a:return 0
    if nim>b:return 1
    if nim>c:return 2
    return 3


def recognize_rate_stage(panel: pd.DataFrame,yc_gov: pd.DataFrame,cfg: StageConfig) -> pd.DataFrame:
    y10=(yc_gov[yc_gov["maturity_year"]==10][["period","yld_pct"]]
         .drop_duplicates("period").set_index("period")["yld_pct"])
    df=panel[["period","LPR1_pct","DR007_pct","NIM_state_pct"]].copy()
    df["Y10_pct"]=df["period"].astype(str).map(y10).astype(float)
    raw=[]; lpr_s=[]; dr_s=[]; y10_s=[]
    w=np.asarray(cfg.weights,float); w=w/w.sum()
    for _,r in df.iterrows():
        sl=_nearest_stage(r["LPR1_pct"],cfg.lpr_centres); sd=_nearest_stage(r["DR007_pct"],cfg.dr_centres); sy=_nearest_stage(r["Y10_pct"],cfg.y10_centres)
        score=w[0]*sl+w[1]*sd+w[2]*sy
        raw.append(int(np.clip(np.rint(score),0,len(cfg.lpr_centres)-1))); lpr_s.append(sl); dr_s.append(sd); y10_s.append(sy)
    filt=_persistence_filter(raw,cfg.persist_k)
    nim_state=np.array([_nim_state_index(x,cfg.nim_thresholds) for x in df["NIM_state_pct"]])
    expected=np.clip(filt-1,0,3)
    divergence=nim_state-expected
    internal=np.where(divergence>cfg.divergence_tolerance,"PressureAhead",
                      np.where(divergence < -cfg.divergence_tolerance,"BufferAvailable","Matched"))
    # 3-phase labels used by A2/A4, based on the primary LPR anchor.
    lpr=df["LPR1_pct"].to_numpy(float)
    phase=np.where(lpr>=2.75,"Enter",np.where(lpr>=1.75,"Visible","Deep"))
    dr=df["DR007_pct"].to_numpy(float); yv=df["Y10_pct"].to_numpy(float)
    drvol=pd.Series(dr).rolling(4,min_periods=3).std().to_numpy(); yvol=pd.Series(yv).rolling(4,min_periods=3).std().to_numpy()
    scenarios=[]
    for t in range(len(df)):
        sc="Stable"
        if t>=2 and lpr[t]<lpr[t-1]<lpr[t-2] and yv[t]<yv[t-1]: sc="Downward"
        if t>=4:
            prev_dr=np.nanmean(drvol[max(0,t-4):t]); prev_y=np.nanmean(yvol[max(0,t-4):t])
            if (np.isfinite(drvol[t]) and drvol[t]>1.25*prev_dr) or (np.isfinite(yvol[t]) and yvol[t]>1.25*prev_y): sc="Volatility"
        if t>0 and filt[t]>filt[t-1] and nim_state[t]>nim_state[t-1]: sc="PressureSwitch"
        scenarios.append(sc)
    # Persistence confirmation for a non-stable scenario.
    switch=[]
    for t,sc in enumerate(scenarios):
        if sc=="Stable": switch.append(False); continue
        lo=max(0,t-cfg.persist_k+1); switch.append(all(x==sc for x in scenarios[lo:t+1]) and t-lo+1>=cfg.persist_k)
    labels=["4pct_or_above","3pct","2pct","1pct","0pct","negative"]
    out=df.copy()
    out["Stage_LPR"]=lpr_s; out["Stage_DR007"]=dr_s; out["Stage_Y10"]=y10_s; out["RateStageIndexRaw"]=raw; out["RateStageIndex"]=filt
    out["RateStage"]=[labels[min(i,len(labels)-1)] for i in filt]; out["LowRatePhase"]=phase
    out["NIMStateIndex"]=nim_state; out["ExpectedNIMStateIndex"]=expected; out["Divergence"]=divergence; out["InternalFlag"]=internal
    out["Scenario"]=scenarios; out["SwitchFlag"]=switch; out["DR007Vol4Q"]=drvol; out["Y10Vol4Q"]=yvol
    return out

# 附录编号统一入口
run_a7 = recognize_rate_stage
