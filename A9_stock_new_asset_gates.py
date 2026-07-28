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

"""A9 存量资产分层与新增资源准入算法

与附录算法编号一一对应。主要入口：stock_and_new_asset_gates。
"""

def stock_and_new_asset_gates(a8: pd.DataFrame,params: pd.DataFrame,watch_history: dict[str,int]|None=None) -> pd.DataFrame:
    hist=watch_history or {}; pp=params.set_index("bucket").loc[BUCKETS]
    med_rorac=float(a8["RORAC"].median()); med_rorwa=float(a8["RORWA"].median()); med_dir=float(a8["DirectionIndex"].median())
    rows=[]
    for _,r in a8.iterrows():
        b=r["bucket"]; p=pp.loc[b]
        cash_stable=(b not in {"trading"}); return_ok=r["NetRAR"]>0; risk_control=float(p["haircut"])<0.10
        customer_strong=r["CustomerValue"]>=a8["CustomerValue"].median(); repair=(r["NetRAR"]>-abs(a8["NetRAR"].median()) and risk_control)
        nsfr_high=float(p["rsf"])>0.65; migration_rising=float(p["haircut"])>=0.10
        if cash_stable and return_ok and risk_control and customer_strong: pool="StableHold"
        elif repair: pool="RevalueOptimize"
        elif (not return_ok) or nsfr_high or migration_rising: pool="ReduceReplace"
        else: pool="ExitDispose"
        if r["DirectionIndex"]>=med_dir and r["Action"]=="IncreaseOrMaintain": tier="Priority"
        elif r["NetRAR"]>0 and risk_control: tier="Watch"
        else:tier="ReduceExit"
        if tier=="Watch" and hist.get(b,0)>=2:tier="ReduceExit"
        if tier=="ReduceExit" and r["RORAC"]>=med_rorac and r["RORWA"]>=med_rorwa and customer_strong:tier="Watch"
        gates={
            "Gate1_Return":bool(return_ok),
            "Gate2_Capital":bool(r["RORAC"]>=med_rorac*0.8 and r["RORWA"]>=med_rorwa*0.8),
            "Gate3_Liquidity":bool(float(p["rsf"])<=0.70 and float(p["duration_year"])<=5.0),
            "Gate4_Risk":bool(float(p["haircut"])<=0.12),
            "Gate5_Customer":bool(r["CustomerValue"]>=0.5*a8["CustomerValue"].median()),
        }
        if all(gates.values()): action="ApproveAndAllocate"
        elif not gates["Gate1_Return"]: action="RepriceOrReduceLimit"
        elif not gates["Gate2_Capital"]: action="ShiftToCapitalEfficientAsset"
        elif not gates["Gate3_Liquidity"]: action="ShortenMaturityOrAddStableFunding"
        elif not gates["Gate4_Risk"]: action="WatchLimitOrReject"
        else: action="AdjustServicePackageOrLowerPriority"
        rows.append({"bucket":b,"bucket_cn":BUCKET_CN[b],"StockPool":pool,"TierList":tier,**gates,"Action":action})
    return pd.DataFrame(rows)

# 附录编号统一入口
run_a9 = stock_and_new_asset_gates
