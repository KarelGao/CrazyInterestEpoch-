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

"""A10 回测、触发与滚动校准算法

与附录算法编号一一对应。主要入口：run_backtest_and_triggers。
"""

def _eval_trigger(value: float,prev: float|None,rule: dict[str,Any]) -> bool:
    typ=rule.get("type"); thr=float(rule.get("value",0))
    if not np.isfinite(value): return False
    if typ=="min":return value<thr
    if typ=="max":return value>thr
    if prev is None or not np.isfinite(prev):return False
    change=value-prev
    if typ=="abs_change_ge":return abs(change)>=thr
    if typ=="change_le":return change<=thr
    if typ=="change_ge":return change>=thr
    return False


def evaluate_trigger_group(signals: dict[str,float],prev_signals: dict[str,float]|None,rules: dict[str,dict[str,Any]]) -> dict[str,bool]:
    prev_signals=prev_signals or {}
    return {name:_eval_trigger(float(signals.get(name,np.nan)),float(prev_signals.get(name,np.nan)) if name in prev_signals else None,rule)
            for name,rule in rules.items()}


def aggregate_severity(*flags: dict[str,bool]) -> str:
    n=sum(sum(bool(v) for v in f.values()) for f in flags); total=sum(len(f) for f in flags)
    if n==0:return "正常"
    if n==1:return "关注"
    if n<=max(2,total//3):return "预警"
    return "压力"


def run_backtest_and_triggers(panel: pd.DataFrame,allocation: pd.DataFrame,stage: pd.DataFrame,params: pd.DataFrame,
                              trigger_set: dict[str,dict[str,dict[str,Any]]]) -> dict[str,Any]:
    pp=params.set_index("bucket").loc[BUCKETS]
    omega=pp["omega_rwa_density"].to_numpy(float); hqla=pp["hqla_ratio"].to_numpy(float); rsf=pp["rsf"].to_numpy(float)
    rows=[]; trig_rows=[]; prev_ext=prev_op=prev_con=None
    for t in range(len(panel)):
        r=panel.iloc[t]; wplan=allocation[[f"w_mpc_{b}" for b in BUCKETS]].iloc[t].to_numpy(float); wact=r[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
        A=float(r["A_total_assets_rmb_bn"])
        plan_metrics={"Return":float(np.mean([allocation.iloc[t][f"mu_{b}"] for b in BUCKETS])),
                      "Capital":float(omega@wplan),"Liquidity":float(hqla@wplan),"StableFunding":float(rsf@wplan),
                      "Risk":float(pp["haircut"].to_numpy(float)@wplan),"Valuation":float(pp["duration_year"].to_numpy(float)@wplan),
                      "Collaboration":float(wplan[5]),"CustomerValue":float(wplan[0]+wplan[5])}
        act_metrics={"Return":float(r["NIM_state_pct"])/100,"Capital":float(omega@wact),"Liquidity":float(hqla@wact),"StableFunding":float(rsf@wact),
                     "Risk":float(pp["haircut"].to_numpy(float)@wact),"Valuation":float(pp["duration_year"].to_numpy(float)@wact),
                     "Collaboration":float(wact[5]),"CustomerValue":float(wact[0]+wact[5])}
        # Attribution is rule-based and auditable, not a causal decomposition claim.
        rate_change=float(r["LPR1_pct"]-panel.iloc[t-1]["LPR1_pct"]) if t else 0.0
        cap_tight=float(r["CAR_pct"]-panel.iloc[t-1]["CAR_pct"]) if t else 0.0
        liq_tight=float(r["NSFR_pct"]-panel.iloc[t-1]["NSFR_pct"]) if t else 0.0
        attr="Execution"
        if abs(rate_change)>=max(abs(cap_tight)/10,abs(liq_tight)/10):attr="RateChange"
        elif cap_tight<0:attr="CapitalTightening"
        elif liq_tight<0:attr="LiquidityTightening"
        for d in plan_metrics:
            rows.append({"period":r["period"],"Dimension":d,"Plan":plan_metrics[d],"Actual":act_metrics[d],"Deviation":act_metrics[d]-plan_metrics[d],"Attribution":attr})

        st=stage.iloc[t]
        ext={"LPR1_change":float(r["LPR1_pct"]),"DR007_change":float(r["DR007_pct"]),"Y10_change":float(st["Y10_pct"])}
        op={"NIM":float(r["NIM_state_pct"]),"NIM_change":float(r["NIM_state_pct"])}
        con={"CAR":float(r["CAR_pct"]),"LCR":float(r["LCR_pct"]),"NSFR":float(r["NSFR_pct"])}
        ef=evaluate_trigger_group(ext,prev_ext,trigger_set.get("External",{})); of=evaluate_trigger_group(op,prev_op,trigger_set.get("Operating",{})); cf=evaluate_trigger_group(con,prev_con,trigger_set.get("Constraint",{}))
        level=aggregate_severity(ef,of,cf)
        actions=[]
        if of.get("NIM",False) or of.get("NIM_change",False):actions.append("压降低效资产，增加稳定票息与服务收入贡献")
        if cf.get("CAR",False):actions.append("提高RORAC门槛并控制高RWA资产")
        if cf.get("NSFR",False):actions.append("控制长久期资产并补充稳定资金/HQLA")
        if ef.get("DR007_change",False) or ef.get("Y10_change",False):actions.append("缩短久期、降低方向性暴露并复核套保")
        if not actions and level!="正常":actions.append("复核配置带宽、FTP与风险预算")
        updates=[]
        if level in {"预警","压力"}:updates=["Update HMM transition/state probabilities","Update Bayesian/Kalman parameters","Recalibrate quantile forecasts","Re-solve MPC","Retune RL penalties within bounds"]
        trig_rows.append({"period":r["period"],"ExternalFlags":json.dumps(ef,ensure_ascii=False),"OperatingFlags":json.dumps(of,ensure_ascii=False),"ConstraintFlags":json.dumps(cf,ensure_ascii=False),
                          "TriggerLevel":level,"CorrectionAction":"；".join(actions),"ParameterUpdate":"；".join(updates)})
        prev_ext,prev_op,prev_con=ext,op,con
    return {"BacktestTable":pd.DataFrame(rows),"TriggerTable":pd.DataFrame(trig_rows),
            "LatestTriggerLevel":trig_rows[-1]["TriggerLevel"] if trig_rows else "正常",
            "LatestCorrectionAction":trig_rows[-1]["CorrectionAction"] if trig_rows else "",
            "LatestParameterUpdate":trig_rows[-1]["ParameterUpdate"] if trig_rows else ""}

# 附录编号统一入口
run_a10 = run_backtest_and_triggers
