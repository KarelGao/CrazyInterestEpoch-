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

"""A5 利率下降情景下盈利弹性测算算法

与附录算法编号一一对应。主要入口：profitability_elasticity。
"""

def reprice_share(bucket: dict[str, Any], horizon_years: float) -> float:
    freq = max(float(bucket.get("RepriceFreq", 1.0)), 1e-6)
    mat = max(float(bucket.get("RemainMat", 1.0)), 1e-6)
    prepay = max(float(bucket.get("PrepayRate", 0.0)), 0.0)
    eff_mat = max(1.0/freq, mat)
    share = 1.0 - np.exp(-horizon_years/eff_mat)
    return float(np.clip(share + prepay*horizon_years, 0, 1))


def profitability_elasticity(
    delta_r_set: Sequence[float], balance_sheet0: dict[str, float],
    bucket_a: list[dict[str, Any]], bucket_l: list[dict[str, Any]],
    beta_range_a: tuple[float,float] = (0.65, 0.95), beta_range_l: tuple[float,float] = (0.20, 0.60),
    horizon_years: float = 1.0, tax_rate: float = 0.25, capital_rule: str = "EquityFixed",
) -> dict[str, Any]:
    A0=float(balance_sheet0["TotalAssets"]); E0=float(balance_sheet0["Equity"])
    avg_ea=float(balance_sheet0.get("AvgEarningAssets", A0)); profit0=float(balance_sheet0.get("Profit0",0.0))
    roe0=float(balance_sheet0.get("ROE", profit0/max(E0,1e-12)))
    payout=float(balance_sheet0.get("PayoutRatio",0.30))
    scenarios=[]; details=[]
    cases=[("Low",beta_range_a[0],beta_range_l[0]),
           ("Base",np.mean(beta_range_a),np.mean(beta_range_l)),
           ("High",beta_range_a[1],beta_range_l[1])]
    for dr in delta_r_set:
        for tag, ba, bl in cases:
            asset_detail=[]; liab_detail=[]
            dint_income=0.0; dint_expense=0.0
            for b in bucket_a:
                sh=reprice_share(b,horizon_years); dy=ba*dr*sh; c=float(b["Balance"])*dy
                dint_income += c
                asset_detail.append({"Bucket":b.get("Name","asset"),"Balance":b["Balance"],"Share":sh,"Beta":ba,"DeltaYield":dy,"DeltaIntIncome":c})
            for b in bucket_l:
                sh=reprice_share(b,horizon_years); raw=bl*dr*sh
                base=float(b.get("CostRate0",0.0)); floor=float(b.get("CostFloor",0.0))
                new=max(base+raw,floor); dc=new-base; c=float(b["Balance"])*dc
                dint_expense += c
                liab_detail.append({"Bucket":b.get("Name","liability"),"Balance":b["Balance"],"Share":sh,"Beta":bl,"CostFloor":floor,"DeltaCost":dc,"DeltaIntExpense":c})
            dnii=dint_income-dint_expense; dprofit=dnii*(1-tax_rate)
            dnim=dnii/max(avg_ea,1e-12); droa=dprofit/max(A0,1e-12)
            if capital_rule == "EquityFixed":
                droe=dprofit/max(E0,1e-12)
            else:
                E1=E0+dprofit*(1-payout)
                droe=(profit0+dprofit)/max(E1,1e-12)-roe0
            scenarios.append({"DeltaRate":dr,"Case":tag,"DeltaNII":dnii,"DeltaNIM":dnim,"DeltaROA":droa,"DeltaROE":droe})
            details.append({"DeltaRate":dr,"Case":tag,"AssetDetail":asset_detail,"LiabDetail":liab_detail})
    scen=pd.DataFrame(scenarios)
    plot=scen[scen["Case"]=="Base"][["DeltaRate","DeltaNIM","DeltaROE"]].reset_index(drop=True)
    return {"ScenarioTable":scen,"DetailTable":details,"PlotData":plot}

# 附录编号统一入口
run_a5 = profitability_elasticity
