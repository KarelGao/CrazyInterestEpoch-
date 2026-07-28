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

"""A8 RORAC / RORWA 风险调整资本效率诊断算法

与附录算法编号一一对应。主要入口：risk_adjusted_diagnostics。
"""

def risk_adjusted_diagnostics(panel: pd.DataFrame,params: pd.DataFrame,yields: pd.DataFrame,
                              allocation_history: pd.DataFrame|None=None,econ_capital_ratio: float=0.105) -> pd.DataFrame:
    row=panel.iloc[-1]; A=float(row["A_total_assets_rmb_bn"]); rL=float(row["rL_avg_interest_bearing_cost_pct"])/100
    pp=params.set_index("bucket").loc[BUCKETS]
    if allocation_history is not None and len(allocation_history):
        w=np.array([float(allocation_history.iloc[-1][f"w_mpc_{b}"]) for b in BUCKETS])
    else:w=row[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
    yld=yields.iloc[-1].to_numpy(float)/100
    total_op=max(float(row.get("operating_cost_proxy_rmb_bn",0.0)),0.0)
    # Customer-value proxy uses non-interest income allocated by business affinity; marked Estimated.
    nonint=max(float(row.get("non_interest_income_proxy_rmb_bn",0.0)),0.0)
    customer_affinity=np.array([0.8,0.4,0.3,0.8,0.2,1.0])
    customer=nonint*customer_affinity/customer_affinity.sum()
    rows=[]
    for i,b in enumerate(BUCKETS):
        exposure=A*w[i]; income=exposure*yld[i]; funding=exposure*rL
        op=total_op*(exposure/max(A,1e-12))
        # In absence of disclosed bucket EL, use a transparent haircut-linked conservative proxy.
        el_rate=float(pp.iloc[i]["haircut"])*0.05
        el=exposure*el_rate
        rwa=exposure*float(pp.iloc[i]["omega_rwa_density"])
        net=income-funding-op-el
        # A zero regulatory risk weight does not imply infinite economic return efficiency.
        # Leave RORWA/RORAC undefined unless a positive capital denominator is available.
        econ=rwa*econ_capital_ratio if rwa>1e-8 else np.nan
        rorwa=net/rwa if rwa>1e-8 else np.nan
        rorac=net/econ if np.isfinite(econ) and econ>1e-8 else np.nan
        rows.append({"bucket":b,"bucket_cn":BUCKET_CN[b],"Exposure":exposure,"Income":income,"FundingCost":funding,"OpCost":op,"EL":el,
                     "RWA":rwa,"EconCapital":econ,"NetRAR":net,"RORWA":rorwa,"RORAC":rorac,
                     "CapitalIntensity":rwa/max(exposure,1e-8),"LiquidityValue":float(pp.iloc[i]["hqla_ratio"])-float(pp.iloc[i]["rsf"]),
                     "CustomerValue":customer[i],"Duration":float(pp.iloc[i]["duration_year"]),"DataFlag":"Estimated"})
    out=pd.DataFrame(rows)
    # Portfolio-weighted reference values.
    expw=out["Exposure"]/max(out["Exposure"].sum(),1e-12)
    refs={}
    for c in ["RORWA","RORAC","CapitalIntensity"]:
        finite=np.isfinite(out[c].to_numpy(float))
        refs[c]=float(np.sum(expw[finite]*out.loc[finite,c])/max(expw[finite].sum(),1e-12)) if np.any(finite) else np.nan
        ref=refs[c]
        out[c+"Index"]=100*out[c]/ref if np.isfinite(ref) and ref>0 else np.nan
    def pct_rank(s,ascending=True):
        rr=s.rank(pct=True,ascending=ascending)*100
        return rr.fillna(50.0)
    out["DirectionIndex"]=(0.35*pct_rank(out["RORAC"],True)+0.30*pct_rank(out["RORWA"],True)+
                           0.20*pct_rank(out["CapitalIntensity"],False)+0.10*pct_rank(out["LiquidityValue"],True)+0.05*pct_rank(out["CustomerValue"],True))
    med_rorac=out["RORAC"].median(skipna=True); med_rorwa=out["RORWA"].median(skipna=True); med_cap=out["CapitalIntensity"].median(); cap_q75=out["CapitalIntensity"].quantile(0.75)
    actions=[]
    for _,r in out.iterrows():
        if r["NetRAR"]>0 and np.isfinite(r["RORAC"]) and np.isfinite(r["RORWA"]) and r["RORAC"]>=med_rorac and r["RORWA"]>=med_rorwa and r["CapitalIntensity"]<=cap_q75: actions.append("IncreaseOrMaintain")
        elif r["NetRAR"]>0 and (r["CapitalIntensity"]>med_cap or r["LiquidityValue"]<0): actions.append("RepriceOrRestructure")
        else: actions.append("ReduceOrExit")
    out["Action"]=actions
    return out.sort_values("DirectionIndex",ascending=False).reset_index(drop=True)

# 附录编号统一入口
run_a8 = risk_adjusted_diagnostics
