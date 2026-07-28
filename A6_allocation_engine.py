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
from A3_shadow_rate import _interpolate_curve

"""A6 NIM-State Robust 核心资产配置算法

与附录算法编号一一对应。主要入口：run_allocation_engine。
"""

@dataclass
class GaussianHMM:
    n_states: int
    n_dim: int
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray

    def _logpdf_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        T = len(X); out = np.empty((T, self.n_states))
        for k in range(self.n_states):
            S = (self.Sigma[k] + self.Sigma[k].T)/2 + 1e-8*np.eye(self.n_dim)
            sign, logdet = np.linalg.slogdet(S)
            inv = np.linalg.pinv(S)
            xc = X - self.mu[k]
            q = np.einsum("ti,ij,tj->t", xc, inv, xc)
            out[:,k] = -0.5*(self.n_dim*np.log(2*np.pi) + logdet + q)
        return out

    def forward_backward(self, X: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
        X=np.asarray(X,float); T=len(X); K=self.n_states
        logB=self._logpdf_matrix(X)
        logA=np.log(np.clip(self.A,1e-15,None)); logpi=np.log(np.clip(self.pi,1e-15,None))
        la=np.empty((T,K)); lb=np.zeros((T,K))
        la[0]=logpi+logB[0]
        for t in range(1,T):
            la[t]=logB[t]+logsumexp(la[t-1][:,None]+logA,axis=0)
        ll=float(logsumexp(la[-1]))
        for t in range(T-2,-1,-1):
            lb[t]=logsumexp(logA+logB[t+1][None,:]+lb[t+1][None,:],axis=1)
        lg=la+lb-ll
        gamma=np.exp(lg); gamma/=np.clip(gamma.sum(axis=1,keepdims=True),1e-15,None)
        xi=np.empty((max(T-1,0),K,K))
        for t in range(T-1):
            lx=la[t][:,None]+logA+logB[t+1][None,:]+lb[t+1][None,:]-ll
            x=np.exp(lx); xi[t]=x/np.clip(x.sum(),1e-15,None)
        return gamma,xi,ll

    def em_train(self, X: np.ndarray, n_iter: int=60, reg: float=1e-4, tol: float=1e-6) -> float:
        X=np.asarray(X,float); last=-np.inf
        for _ in range(n_iter):
            gamma,xi,ll=self.forward_backward(X)
            self.pi=np.clip(gamma[0],1e-12,None); self.pi/=self.pi.sum()
            counts=xi.sum(axis=0)+1e-8
            self.A=counts/counts.sum(axis=1,keepdims=True)
            for k in range(self.n_states):
                w=gamma[:,k]; sw=max(w.sum(),1e-12)
                m=(w[:,None]*X).sum(axis=0)/sw
                xc=X-m
                S=np.einsum("t,ti,tj->ij",w,xc,xc)/sw + reg*np.eye(self.n_dim)
                self.mu[k]=m; self.Sigma[k]=S
            if np.isfinite(last) and abs(ll-last)<tol: break
            last=ll
        return float(last)


def init_hmm(X: np.ndarray, n_states: int=3, seed: int=0) -> GaussianHMM:
    rng=np.random.default_rng(seed); X=np.asarray(X,float); T,d=X.shape
    # deterministic-ish quantile centroids along first principal component for stable starts
    Xc=X-X.mean(axis=0); u,s,v=np.linalg.svd(Xc,full_matrices=False)
    score=Xc@v[0] if len(v) else np.arange(T)
    qs=np.quantile(score,np.linspace(0.15,0.85,n_states))
    idx=[int(np.argmin(abs(score-q))) for q in qs]
    mu=X[idx].copy()
    base=np.cov(X.T) if T>2 else np.eye(d)
    if np.ndim(base)==0: base=np.eye(d)*float(base)
    Sigma=np.array([base+1e-3*np.eye(d) for _ in range(n_states)])
    A=np.full((n_states,n_states),0.10/max(n_states-1,1))
    np.fill_diagonal(A,0.90 if n_states>1 else 1.0)
    A=A/A.sum(axis=1,keepdims=True)
    return GaussianHMM(n_states,d,np.ones(n_states)/n_states,A,mu,Sigma)


def bayes_transition_mean(xi: np.ndarray, alpha0: float=1.0) -> np.ndarray:
    if len(xi)==0: return np.ones((1,1))
    counts=xi.sum(axis=0)+alpha0
    return counts/counts.sum(axis=1,keepdims=True)


def build_hmm_features(panel: pd.DataFrame, params: pd.DataFrame) -> tuple[np.ndarray,list[str]]:
    df=panel.copy()
    wcols=[f"w_{b}" for b in BUCKETS]
    omega=params.set_index("bucket").loc[BUCKETS,"omega_rwa_density"].to_numpy(float)
    W=df[wcols].to_numpy(float)
    rwa_int=W@omega
    dw=np.r_[0.0,np.sum(np.abs(np.diff(W,axis=0)),axis=1)]
    feats=np.column_stack([
        df["NIM_state_pct"].to_numpy(float),
        np.r_[0.0,np.diff(df["NIM_state_pct"].to_numpy(float))],
        df["CAR_pct"].to_numpy(float),
        df["LCR_pct"].to_numpy(float),
        df["NSFR_pct"].to_numpy(float),
        np.r_[0.0,np.diff(rwa_int)],
        dw,
    ])
    names=["NIM","dNIM","CAR","LCR","NSFR","dRWAint","dW"]
    return feats,names


def map_hmm_regimes(hmm: GaussianHMM, scaler: StandardScaler, feature_names: list[str]) -> dict[int,str]:
    # Convert state means back to original units, then construct a stress score.
    orig=scaler.inverse_transform(hmm.mu)
    idx={n:i for i,n in enumerate(feature_names)}
    # Standardise state means across states for score so units do not dominate.
    z=(orig-orig.mean(axis=0))/np.where(orig.std(axis=0)>1e-9,orig.std(axis=0),1.0)
    stress=(-z[:,idx["NIM"]]-0.7*z[:,idx["CAR"]]-0.35*z[:,idx["LCR"]]-0.45*z[:,idx["NSFR"]]
            +0.25*z[:,idx["dRWAint"]]+0.20*z[:,idx["dW"]])
    order=np.argsort(stress) # low stress -> controllable
    labels=["controllable","explicit","hardening"]
    return {int(k):labels[min(i,len(labels)-1)] for i,k in enumerate(order)}


def dynamic_bayesian_nim_regression(panel: pd.DataFrame, regime_hard_prob: np.ndarray | None=None) -> dict[str,Any]:
    """Kalman dynamic linear regression for drifting repricing/stickiness parameters."""
    y=panel["NIM_state_pct"].to_numpy(float)
    X=np.column_stack([
        np.ones(len(panel)),
        panel["LPR1_pct"].to_numpy(float),
        panel["DR007_pct"].to_numpy(float),
        panel["r_dep_deposit_cost_proxy_pct"].to_numpy(float),
        panel["alpha_mkt_funding_share_proxy"].to_numpy(float),
    ])
    d=X.shape[1]; theta=np.linalg.lstsq(X,y,rcond=None)[0]; P=np.eye(d)*5.0
    R=max(np.var(y-X@theta),1e-3); base_q=2e-4
    means=[]; covs=[]; pred=[]; pred_var=[]
    for t in range(len(y)):
        hp=float(regime_hard_prob[t]) if regime_hard_prob is not None else 0.0
        Q=np.eye(d)*base_q*(1+4*hp)
        Pp=P+Q; tp=theta
        x=X[t]
        yp=float(x@tp); S=float(x@Pp@x+R)
        K=(Pp@x)/max(S,1e-12)
        theta=tp+K*(y[t]-yp); P=(np.eye(d)-np.outer(K,x))@Pp
        means.append(theta.copy()); covs.append(P.copy())
        # one-step predictive using current regressors as random-walk proxy
        pred.append(float(x@theta)); pred_var.append(float(x@P@x+R))
    return {"theta":np.asarray(means),"P":np.asarray(covs),"pred_mean":np.asarray(pred),
            "pred_var":np.asarray(pred_var),"feature_names":["const","LPR1","DR007","r_dep","alpha"],"R":R}


def _curve_factors(yc_gov: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    piv=yc_gov.pivot(index="period",columns="maturity_year",values="yld_pct")
    out=pd.DataFrame(index=panel["period"].astype(str))
    mats=np.array(piv.columns,float)
    for p in out.index:
        row=piv.loc[p].to_numpy(float) if p in piv.index else np.full(len(mats),np.nan)
        row=_interpolate_curve(row,mats)
        out.loc[p,"curve_level"]=float(np.mean(row))
        out.loc[p,"curve_slope"]=float(row[-1]-row[np.argmin(abs(mats-1.0))])
        out.loc[p,"curve_curvature"]=float(2*row[np.argmin(abs(mats-5.0))]-row[0]-row[-1])
    return out.reset_index(drop=True)


def build_forecast_features(panel: pd.DataFrame, yc_gov: pd.DataFrame, gamma: np.ndarray | None=None) -> tuple[np.ndarray,list[str]]:
    cf=_curve_factors(yc_gov,panel)
    cols=["LPR1_pct","LPR5_pct","DR007_pct","alpha_mkt_funding_share_proxy","g_asset_growth_q"]
    X=[panel[c].to_numpy(float) for c in cols]
    names=cols.copy()
    for c in ["curve_level","curve_slope","curve_curvature"]:
        X.append(cf[c].to_numpy(float)); names.append(c)
    for b in BUCKETS:
        X.append(panel[f"w_{b}"].to_numpy(float)); names.append(f"w_{b}")
    if gamma is not None:
        for j in range(gamma.shape[1]):
            X.append(gamma[:,j]); names.append(f"regime_p{j}")
    return np.column_stack(X),names


def quantile_forecast_latest(
    panel: pd.DataFrame, yc_gov: pd.DataFrame, gamma: np.ndarray | None=None,
    quantiles: tuple[float,...]=(0.10,0.50,0.90), seed: int=0,
) -> dict[str,dict[float,float]]:
    X,_=build_forecast_features(panel,yc_gov,gamma)
    targets=["NIM_state_pct","CAR_pct","LCR_pct","NSFR_pct"]
    out={}
    if len(panel)<12: return out
    Xtr=X[:-1]; xlast=X[-1:]
    for target in targets:
        y=panel[target].to_numpy(float)[1:] # t target from t-1 features
        Xuse=Xtr[:len(y)]
        pred={}
        for q in quantiles:
            mdl=GradientBoostingRegressor(loss="quantile",alpha=q,n_estimators=150,max_depth=2,
                                          learning_rate=0.04,random_state=seed)
            mdl.fit(Xuse,y)
            pred[q]=float(mdl.predict(xlast)[0])
        out[target]=pred
    return out


def derive_bucket_yields(panel: pd.DataFrame, yc_gov: pd.DataFrame, yc_cred: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the six bucket yields from the same economic logic as main.py."""
    g=yc_gov.pivot(index="period",columns="maturity_year",values="yld_pct")
    c=yc_cred.pivot(index="period",columns="maturity_year",values="yld_pct")
    rows=[]
    for _,r in panel.iterrows():
        p=str(r["period"])
        def get(piv,m):
            mats=np.array(piv.columns,float); row=_interpolate_curve(piv.loc[p].to_numpy(float),mats)
            return float(np.interp(m,mats,row))
        gov5,gov10,cred3,cred5=get(g,5),get(g,10),get(c,3),get(c,5)
        L1=float(r["LPR1_pct"]); L5=float(r["LPR5_pct"]); dr=float(r["DR007_pct"])
        vals={
            "credit":0.92*L1+0.10*L5-0.25,
            "bond":0.55*cred5+0.45*gov10+0.15,
            "ib":dr+0.20,
            "trading":0.55*cred3+0.25*dr+0.20*gov5+0.10,
            "cb_gov":0.80*gov5+0.10,
            "wm_fund":0.55*cred5+0.25*L1+0.20*gov5+0.35,
        }
        rows.append(vals)
    return pd.DataFrame(rows,index=panel.index)


@dataclass
class RegulatoryMap:
    A_total: float
    capital_amt: float
    nco_amt: float
    asf_amt: float
    omega: np.ndarray
    hqla: np.ndarray
    rsf: np.ndarray
    duration: np.ndarray
    target_sum: float

    @classmethod
    def calibrate(cls,row: pd.Series,w_anchor: np.ndarray,params: pd.DataFrame,target_sum: float | None = None):
        pp=params.set_index("bucket").loc[BUCKETS]
        omega=pp["omega_rwa_density"].to_numpy(float)
        hqla=pp["hqla_ratio"].to_numpy(float)
        rsf=pp["rsf"].to_numpy(float)
        duration=pp["duration_year"].to_numpy(float)
        A=float(row["A_total_assets_rmb_bn"]); target=float(np.sum(w_anchor) if target_sum is None else target_sum)
        rwa=max(A*(omega@w_anchor),1e-8)
        cap=float(row["CAR_pct"])/100*rwa
        hq_amt=max(A*(hqla@w_anchor),1e-8)
        nco=hq_amt/max(float(row["LCR_pct"])/100,1e-6)
        rsf_amt=max(A*(rsf@w_anchor),1e-8)
        asf=float(row["NSFR_pct"])/100*rsf_amt
        return cls(A,cap,nco,asf,omega,hqla,rsf,duration,target)

    def metrics(self,w: np.ndarray) -> dict[str,float]:
        w=np.asarray(w,float)
        rwa=max(self.A_total*(self.omega@w),1e-10)
        h=max(self.A_total*(self.hqla@w),1e-10)
        rs=max(self.A_total*(self.rsf@w),1e-10)
        return {
            "CAR":100*self.capital_amt/rwa,
            "LCR":100*h/max(self.nco_amt,1e-10),
            "NSFR":100*self.asf_amt/rs,
            "duration":float((self.duration@w)/max(np.sum(w),1e-10)),
            "RWA_intensity":float(self.omega@w),
            "HQLA_ratio":float(self.hqla@w),
            "RSF_intensity":float(self.rsf@w),
        }


def effective_constraint_floors(row: pd.Series,cfg: AllocationConfig,qforecast: dict[str,dict[float,float]]|None=None) -> dict[str,float]:
    floors={"CAR":cfg.car_min,"LCR":cfg.lcr_min,"NSFR":cfg.nsfr_min}
    if not qforecast: return floors
    mapping={"CAR":"CAR_pct","LCR":"LCR_pct","NSFR":"NSFR_pct"}
    for key,col in mapping.items():
        if col in qforecast and cfg.q_delta in qforecast[col]:
            qlow=float(qforecast[col][cfg.q_delta]); current=float(row[col])
            uncertainty=max(0.0,current-qlow)
            floors[key]+=uncertainty
    return floors


def allocation_constraints(reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                           w_lower: np.ndarray,w_upper: np.ndarray) -> list[dict[str,Any]]:
    cons=[
        {"type":"eq","fun":lambda w: np.sum(w)-reg.target_sum},
        {"type":"ineq","fun":lambda w: reg.metrics(w)["CAR"]-floors["CAR"]},
        {"type":"ineq","fun":lambda w: reg.metrics(w)["LCR"]-floors["LCR"]},
        {"type":"ineq","fun":lambda w: reg.metrics(w)["NSFR"]-floors["NSFR"]},
        {"type":"ineq","fun":lambda w: cfg.duration_cap-reg.metrics(w)["duration"]},
        {"type":"ineq","fun":lambda w: cfg.trading_cap-w[3]},
    ]
    return cons


def feasible_slacks(w: np.ndarray,reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                    w_lower: np.ndarray,w_upper: np.ndarray) -> dict[str,float]:
    m=reg.metrics(w)
    d={
        "CAR":m["CAR"]-floors["CAR"],"LCR":m["LCR"]-floors["LCR"],"NSFR":m["NSFR"]-floors["NSFR"],
        "duration":cfg.duration_cap-m["duration"],"trading":cfg.trading_cap-w[3],
        "lower_min":float(np.min(w-w_lower)),"upper_min":float(np.min(w_upper-w)),
        "sum_error":float(abs(np.sum(w)-reg.target_sum)),
    }
    return d


def is_feasible(w: np.ndarray,reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                w_lower: np.ndarray,w_upper: np.ndarray,tol: float=1e-6) -> bool:
    s=feasible_slacks(w,reg,cfg,floors,w_lower,w_upper)
    return (min(s[k] for k in ["CAR","LCR","NSFR","duration","trading","lower_min","upper_min"])>=-tol
            and s["sum_error"]<=1e-5)


def project_to_feasible(w0: np.ndarray,reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                        w_lower: np.ndarray,w_upper: np.ndarray) -> tuple[np.ndarray,str]:
    bounds=list(zip(w_lower,w_upper)); cons=allocation_constraints(reg,cfg,floors,w_lower,w_upper)
    x0=np.clip(w0,w_lower,w_upper)
    x0=project_simplex(x0,reg.target_sum)
    # projection can violate bounds after simplex; use SLSQP from clipped normalized point.
    res=minimize(lambda w: float(np.sum((w-w0)**2)),x0,method="SLSQP",bounds=bounds,constraints=cons,
                 options={"maxiter":cfg.solver_maxiter,"ftol":1e-11,"disp":False})
    if res.success and is_feasible(res.x,reg,cfg,floors,w_lower,w_upper,1e-4): return res.x,"projected"
    # fallback minimise squared violations + distance
    def penalty(w):
        m=reg.metrics(w)
        v=[max(0,floors["CAR"]-m["CAR"]),max(0,floors["LCR"]-m["LCR"]),max(0,floors["NSFR"]-m["NSFR"]),
           max(0,m["duration"]-cfg.duration_cap),max(0,w[3]-cfg.trading_cap),abs(np.sum(w)-reg.target_sum)]
        return np.sum((w-w0)**2)+1e4*np.sum(np.square(v))
    res2=minimize(penalty,x0,method="SLSQP",bounds=bounds,options={"maxiter":cfg.solver_maxiter,"ftol":1e-11})
    return np.asarray(res2.x,float),("projected_penalty" if res2.success else "projection_failed")


def portfolio_utility(w: np.ndarray,mu: np.ndarray,Sigma: np.ndarray,w_prev: np.ndarray,
                      reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                      lambdas: tuple[float,float,float] | None=None) -> float:
    lv,lc,ll=lambdas or (cfg.lambda_var,cfg.lambda_cap,cfg.lambda_liq)
    w=np.asarray(w,float); m=reg.metrics(w)
    ret=float(mu@w); var=float(w@Sigma@w); turn=float(np.sum(np.abs(w-w_prev)))
    cap_cost=cfg.capital_shadow_base*m["RWA_intensity"]
    lcr_soft=floors["LCR"]+cfg.safety_lcr_margin; nsfr_soft=floors["NSFR"]+cfg.safety_nsfr_margin
    liq=(max(0,lcr_soft-m["LCR"])/100)**2*cfg.kappa_lcr+(max(0,nsfr_soft-m["NSFR"])/100)**2*cfg.kappa_nsfr
    car_soft=floors["CAR"]+cfg.safety_car_margin
    cap_near=(max(0,car_soft-m["CAR"])/100)**2
    return ret-lv*var-lc*(cap_cost+cap_near)-ll*liq-cfg.lambda_turn*turn


def solve_weight_utility(w_anchor: np.ndarray,mu: np.ndarray,Sigma: np.ndarray,reg: RegulatoryMap,
                         cfg: AllocationConfig,floors: dict[str,float],w_lower: np.ndarray,w_upper: np.ndarray,
                         lambdas: tuple[float,float,float]|None=None) -> tuple[np.ndarray,dict[str,Any]]:
    anchor,status=project_to_feasible(w_anchor,reg,cfg,floors,w_lower,w_upper)
    cons=allocation_constraints(reg,cfg,floors,w_lower,w_upper); bounds=list(zip(w_lower,w_upper))
    fun=lambda w:-portfolio_utility(w,mu,Sigma,anchor,reg,cfg,floors,lambdas)
    res=minimize(fun,anchor,method="SLSQP",bounds=bounds,constraints=cons,
                 options={"maxiter":cfg.solver_maxiter,"ftol":1e-10,"disp":False})
    w=res.x if res.success and np.all(np.isfinite(res.x)) else anchor
    return np.asarray(w,float),{"status":("optimal" if res.success else f"fallback:{res.message}"),"projection":status,
                                "objective":-float(fun(w)),"metrics":reg.metrics(w),"slacks":feasible_slacks(w,reg,cfg,floors,w_lower,w_upper)}


def solve_incremental_allocation(w_prev: np.ndarray,growth_q: float,mu: np.ndarray,Sigma: np.ndarray,
                                 reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                                 w_lower: np.ndarray,w_upper: np.ndarray,lambdas: tuple[float,float,float]|None=None) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    """Solve new-asset mix x; inherited stock remains in place. Returns (w_new, x_increment, diag)."""
    # Share of current end-period assets represented by current-period net new assets.
    growth=max(float(growth_q),0.0); a=min(max(growth/(1+growth),1e-4),0.25)
    target=reg.target_sum
    x0=project_simplex(np.maximum(w_prev,0),target)
    xb=[(0.0,target) for _ in BUCKETS]
    def w_from_x(x): return (1-a)*w_prev+a*np.asarray(x,float)
    cons=[{"type":"eq","fun":lambda x:np.sum(x)-target}]
    # hard constraints on resulting stock weights + governance bounds
    cons.extend([
        {"type":"ineq","fun":lambda x:reg.metrics(w_from_x(x))["CAR"]-floors["CAR"]},
        {"type":"ineq","fun":lambda x:reg.metrics(w_from_x(x))["LCR"]-floors["LCR"]},
        {"type":"ineq","fun":lambda x:reg.metrics(w_from_x(x))["NSFR"]-floors["NSFR"]},
        {"type":"ineq","fun":lambda x:cfg.duration_cap-reg.metrics(w_from_x(x))["duration"]},
        {"type":"ineq","fun":lambda x:cfg.trading_cap-w_from_x(x)[3]},
    ])
    for i in range(len(BUCKETS)):
        cons.append({"type":"ineq","fun":lambda x,i=i:w_from_x(x)[i]-w_lower[i]})
        cons.append({"type":"ineq","fun":lambda x,i=i:w_upper[i]-w_from_x(x)[i]})
    fun=lambda x:-portfolio_utility(w_from_x(x),mu,Sigma,w_prev,reg,cfg,floors,lambdas)
    res=minimize(fun,x0,method="SLSQP",bounds=xb,constraints=cons,
                 options={"maxiter":cfg.solver_maxiter,"ftol":1e-10,"disp":False})
    if res.success and np.all(np.isfinite(res.x)):
        x=np.asarray(res.x,float); w=w_from_x(x); status="optimal"
    else:
        # fall back to direct feasible weight optimisation, then invert stock-increment equation when possible.
        w,dd=solve_weight_utility(w_prev,mu,Sigma,reg,cfg,floors,w_lower,w_upper,lambdas)
        x=(w-(1-a)*w_prev)/a; x=project_simplex(np.maximum(x,0),target); w=w_from_x(x)
        status=f"fallback_direct:{getattr(res,'message','')};{dd['status']}"
    diag={"status":status,"growth_share":a,"objective":portfolio_utility(w,mu,Sigma,w_prev,reg,cfg,floors,lambdas),
          "metrics":reg.metrics(w),"slacks":feasible_slacks(w,reg,cfg,floors,w_lower,w_upper)}
    return w,x,diag


def solve_mpc_incremental(w_prev: np.ndarray,growth_forecast: Sequence[float],mu_forecast: Sequence[np.ndarray],
                          Sigma: np.ndarray,reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                          w_lower: np.ndarray,w_upper: np.ndarray,lambdas: tuple[float,float,float]|None=None) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    H=min(cfg.mpc_horizon,len(growth_forecast),len(mu_forecast)); n=len(BUCKETS); target=reg.target_sum
    if H<=1: return solve_incremental_allocation(w_prev,growth_forecast[0],mu_forecast[0],Sigma,reg,cfg,floors,w_lower,w_upper,lambdas)
    z0=np.tile(project_simplex(np.maximum(w_prev,0),target),H)
    bounds=[(0,target)]*(H*n)
    def simulate(z):
        w=w_prev.copy(); ws=[]; xs=[]
        for k in range(H):
            x=z[k*n:(k+1)*n]; g=max(float(growth_forecast[k]),0); a=min(max(g/(1+g),1e-4),0.25)
            w=(1-a)*w+a*x; ws.append(w.copy()); xs.append(x.copy())
        return ws,xs
    cons=[]
    for k in range(H):
        cons.append({"type":"eq","fun":lambda z,k=k:np.sum(z[k*n:(k+1)*n])-target})
        for i in range(n):
            cons.append({"type":"ineq","fun":lambda z,k=k,i=i:simulate(z)[0][k][i]-w_lower[i]})
            cons.append({"type":"ineq","fun":lambda z,k=k,i=i:w_upper[i]-simulate(z)[0][k][i]})
        cons += [
            {"type":"ineq","fun":lambda z,k=k:reg.metrics(simulate(z)[0][k])["CAR"]-floors["CAR"]},
            {"type":"ineq","fun":lambda z,k=k:reg.metrics(simulate(z)[0][k])["LCR"]-floors["LCR"]},
            {"type":"ineq","fun":lambda z,k=k:reg.metrics(simulate(z)[0][k])["NSFR"]-floors["NSFR"]},
            {"type":"ineq","fun":lambda z,k=k:cfg.duration_cap-reg.metrics(simulate(z)[0][k])["duration"]},
            {"type":"ineq","fun":lambda z,k=k:cfg.trading_cap-simulate(z)[0][k][3]},
        ]
    def obj(z):
        ws,_=simulate(z); val=0.0; prev=w_prev
        for k,w in enumerate(ws):
            val+=(cfg.discount**k)*portfolio_utility(w,mu_forecast[k],Sigma,prev,reg,cfg,floors,lambdas); prev=w
        return -val
    res=minimize(obj,z0,method="SLSQP",bounds=bounds,constraints=cons,
                 options={"maxiter":cfg.solver_maxiter,"ftol":1e-9,"disp":False})
    if not res.success:
        return solve_incremental_allocation(w_prev,growth_forecast[0],mu_forecast[0],Sigma,reg,cfg,floors,w_lower,w_upper,lambdas)
    ws,xs=simulate(res.x)
    return ws[0],xs[0],{"status":"optimal_mpc","objective":-float(res.fun),"metrics":reg.metrics(ws[0]),
                        "slacks":feasible_slacks(ws[0],reg,cfg,floors,w_lower,w_upper),"growth_share":max(float(growth_forecast[0]),0)/(1+max(float(growth_forecast[0]),0))}


def compute_bandwidth(w_anchor: np.ndarray,reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float],
                      w_lower: np.ndarray,w_upper: np.ndarray) -> pd.DataFrame:
    anchor,_=project_to_feasible(w_anchor,reg,cfg,floors,w_lower,w_upper)
    cons=allocation_constraints(reg,cfg,floors,w_lower,w_upper); bounds=list(zip(w_lower,w_upper)); rows=[]
    for i,b in enumerate(BUCKETS):
        def fmin(w): return w[i]+cfg.bandwidth_rho*np.sum((w-anchor)**2)
        def fmax(w): return -w[i]+cfg.bandwidth_rho*np.sum((w-anchor)**2)
        lo=minimize(fmin,anchor,method="SLSQP",bounds=bounds,constraints=cons,options={"maxiter":cfg.solver_maxiter,"ftol":1e-10})
        hi=minimize(fmax,anchor,method="SLSQP",bounds=bounds,constraints=cons,options={"maxiter":cfg.solver_maxiter,"ftol":1e-10})
        lv=float(lo.x[i]) if lo.success else float(anchor[i]); hv=float(hi.x[i]) if hi.success else float(anchor[i])
        rows.append({"bucket":b,"bucket_cn":BUCKET_CN[b],"anchor":float(anchor[i]),"lower":lv,"upper":hv,"width":max(0,hv-lv),
                     "lower_status":bool(lo.success),"upper_status":bool(hi.success)})
    return pd.DataFrame(rows)


def approximate_shadow_prices(w_star: np.ndarray,mu: np.ndarray,Sigma: np.ndarray,w_prev: np.ndarray,
                              reg: RegulatoryMap,cfg: AllocationConfig,floors: dict[str,float]) -> dict[str,float]:
    """Approximate KKT multipliers for regulatory constraints by local stationarity least squares."""
    util=lambda w:portfolio_utility(w,mu,Sigma,w_prev,reg,cfg,floors)
    gradU=numerical_gradient(util,w_star,1e-5)
    names=["CAR","LCR","NSFR"]
    cols=[]; active=[]
    for name in names:
        def gfun(w,name=name): return floors[name]-reg.metrics(w)[name]
        slack=reg.metrics(w_star)[name]-floors[name]
        # include constraints that are relatively close; far constraints have zero multiplier.
        scale={"CAR":1.0,"LCR":10.0,"NSFR":3.0}[name]
        if slack <= scale:
            cols.append(numerical_gradient(gfun,w_star,1e-5)); active.append(name)
    result={f"mu_{n}":0.0 for n in names}
    if not cols: return result
    # grad U = J_g^T mu + 1*nu; mu>=0, nu free
    M=np.column_stack(cols+[np.ones(len(w_star))])
    lb=np.r_[np.zeros(len(cols)),-np.inf]; ub=np.r_[np.full(len(cols),np.inf),np.inf]
    fit=lsq_linear(M,gradU,bounds=(lb,ub))
    for j,n in enumerate(active): result[f"mu_{n}"]=float(max(fit.x[j],0))
    result["stationarity_resid"]=float(np.linalg.norm(M@fit.x-gradU))
    return result


def boundary_step(w_anchor: np.ndarray,direction: np.ndarray,reg: RegulatoryMap,cfg: AllocationConfig,
                  floors: dict[str,float],w_lower: np.ndarray,w_upper: np.ndarray) -> tuple[float,list[str]]:
    d=np.asarray(direction,float)
    norm=np.linalg.norm(d)
    if norm<1e-12: return 0.0,[]
    d=d/norm
    def ok(eta): return is_feasible(w_anchor+eta*d,reg,cfg,floors,w_lower,w_upper,1e-7)
    lo,hi=0.0,0.05
    while hi<5.0 and ok(hi): lo=hi; hi*=2
    for _ in range(50):
        mid=(lo+hi)/2
        if ok(mid): lo=mid
        else: hi=mid
    w=w_anchor+lo*d; s=feasible_slacks(w,reg,cfg,floors,w_lower,w_upper)
    bind=[]
    tol={"CAR":1e-3,"LCR":1e-2,"NSFR":1e-2,"duration":1e-3,"trading":1e-4,"lower_min":1e-4,"upper_min":1e-4}
    for k in tol:
        if s[k]<=tol[k]: bind.append(k)
    return float(lo),bind


class GovernanceSafeQLearner:
    """Tiny tabular Q-learner. Actions only select penalty multipliers; never asset weights."""
    def __init__(self,base_lambdas: tuple[float,float,float],seed: int=0):
        self.base=np.asarray(base_lambdas,float); self.rng=np.random.default_rng(seed)
        self.actions=np.array([
            [0.7,0.8,0.8],[1.0,1.0,1.0],[1.3,1.0,1.0],[1.0,1.3,1.0],[1.0,1.0,1.3],
            [1.3,1.3,1.0],[1.3,1.0,1.3],[1.0,1.3,1.3],[1.4,1.4,1.4]
        ])
        self.Q={}
    def state(self,p_hard:float,slacks:dict[str,float]) -> tuple[int,int,int,int]:
        r=2 if p_hard>0.6 else (1 if p_hard>0.3 else 0)
        return (r,int(slacks.get("CAR",9)<0.5),int(slacks.get("LCR",99)<10),int(slacks.get("NSFR",99)<3))
    def choose(self,s,eps=0.1):
        q=self.Q.setdefault(s,np.zeros(len(self.actions)))
        if self.rng.random()<eps:return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(q))
    def lambdas(self,a:int)->tuple[float,float,float]:
        return tuple((self.base*self.actions[a]).tolist())
    def update(self,s,a,reward,s2,alpha=0.15,gamma=0.90):
        q=self.Q.setdefault(s,np.zeros(len(self.actions))); q2=self.Q.setdefault(s2,np.zeros(len(self.actions)))
        q[a]+=alpha*(reward+gamma*np.max(q2)-q[a])


def train_penalty_qlearner(panel: pd.DataFrame,params: pd.DataFrame,yields: pd.DataFrame,gamma: np.ndarray,state_map: dict[int,str],
                           cfg: AllocationConfig,episodes:int=4) -> GovernanceSafeQLearner:
    learner=GovernanceSafeQLearner((cfg.lambda_var,cfg.lambda_cap,cfg.lambda_liq),cfg.seed)
    pp=params.set_index("bucket").loc[BUCKETS]
    base_vol=np.array([0.10,0.06,0.03,0.12,0.02,0.09])/100
    Sigma=0.75*np.diag(base_vol**2)+0.25*np.outer(base_vol,base_vol)
    lo=np.asarray(cfg.w_lower,float); hi=np.asarray(cfg.w_upper,float)
    hard_idx=[k for k,v in state_map.items() if v=="hardening"]
    for ep in range(episodes):
        w=panel[[f"w_{b}" for b in BUCKETS]].iloc[0].to_numpy(float)
        for t in range(len(panel)-1):
            row=panel.iloc[t]; w_obs=row[[f"w_{b}" for b in BUCKETS]].to_numpy(float); reg=RegulatoryMap.calibrate(row,w_obs,params,target_sum=float(np.sum(w))); floors={"CAR":cfg.car_min,"LCR":cfg.lcr_min,"NSFR":cfg.nsfr_min}
            sl=feasible_slacks(w,reg,cfg,floors,lo,hi); ph=float(gamma[t,hard_idx].sum()) if hard_idx else 0.0
            s=learner.state(ph,sl); a=learner.choose(s,eps=max(0.05,0.25*(1-ep/max(episodes,1))))
            lam=learner.lambdas(a)
            mu=(yields.iloc[t].to_numpy(float)-float(row["rL_avg_interest_bearing_cost_pct"]))/100
            wn,_,diag=solve_incremental_allocation(w,float(row["g_asset_growth_q"]),mu,Sigma,reg,cfg,floors,lo,hi,lam)
            violation=sum(max(0,-v) for k,v in diag["slacks"].items() if k in {"CAR","LCR","NSFR","duration","trading"})
            reward=float(diag["objective"])-50*violation-0.2*np.sum(abs(wn-w))
            r2=panel.iloc[t+1]; wobs2=r2[[f"w_{b}" for b in BUCKETS]].to_numpy(float); reg2=RegulatoryMap.calibrate(r2,wobs2,params,target_sum=float(np.sum(wn))); sl2=feasible_slacks(wn,reg2,cfg,floors,lo,hi)
            ph2=float(gamma[t+1,hard_idx].sum()) if hard_idx else 0.0; s2=learner.state(ph2,sl2)
            learner.update(s,a,reward,s2); w=wn
    return learner


def run_allocation_engine(panel: pd.DataFrame,yc_gov: pd.DataFrame,yc_cred: pd.DataFrame,params: pd.DataFrame,
                          cfg: AllocationConfig) -> dict[str,Any]:
    feats,fnames=build_hmm_features(panel,params); scaler=StandardScaler(); Xs=scaler.fit_transform(feats)
    hmm=init_hmm(Xs,3,cfg.seed); hmm_ll=hmm.em_train(Xs,50,1e-4)
    gamma,xi,_=hmm.forward_backward(Xs); state_map=map_hmm_regimes(hmm,scaler,fnames)
    A_bayes=bayes_transition_mean(xi,1.0)
    hard_idx=[k for k,v in state_map.items() if v=="hardening"]
    hard_prob=gamma[:,hard_idx].sum(axis=1) if hard_idx else np.zeros(len(panel))
    bayes=dynamic_bayesian_nim_regression(panel,hard_prob) if cfg.use_bayes else None
    qforecast=quantile_forecast_latest(panel,yc_gov,gamma,(cfg.q_delta,0.5,1-cfg.q_delta),cfg.seed) if cfg.use_quantile else {}
    yields=derive_bucket_yields(panel,yc_gov,yc_cred)

    # rolling covariance from reconstructed bucket yields; fallback covariance is always PSD.
    Yret=yields.to_numpy(float)/100
    cov=np.cov(np.diff(Yret,axis=0).T) if len(Yret)>8 else np.diag(np.full(6,1e-4))
    cov=np.nan_to_num(cov,nan=0.0,posinf=0.0,neginf=0.0)+(1e-8*np.eye(6))
    eig=np.linalg.eigvalsh(cov)
    if eig.min()<1e-10: cov+=np.eye(6)*(1e-10-eig.min())
    lo=np.asarray(cfg.w_lower,float); hi=np.asarray(cfg.w_upper,float)

    learner=None
    if cfg.use_rl:
        learner=train_penalty_qlearner(panel,params,yields,gamma,state_map,cfg,episodes=4)

    rows=[]; w=panel[[f"w_{b}" for b in BUCKETS]].iloc[0].to_numpy(float)
    for t in range(len(panel)):
        row=panel.iloc[t]
        w_obs=row[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
        reg=RegulatoryMap.calibrate(row,w_obs,params,target_sum=float(np.sum(w)))
        qf=qforecast if t==len(panel)-1 else None
        floors=effective_constraint_floors(row,cfg,qf)
        # If chance buffer makes current state infeasible beyond repair, revert to statutory/management hard floors.
        test_anchor,_=project_to_feasible(w,reg,cfg,floors,lo,hi)
        if not is_feasible(test_anchor,reg,cfg,floors,lo,hi,1e-3):
            floors={"CAR":cfg.car_min,"LCR":cfg.lcr_min,"NSFR":cfg.nsfr_min}
        # expected net return by bucket, with a small haircut-loss adjustment linked to market volatility.
        mu=(yields.iloc[t].to_numpy(float)-float(row["rL_avg_interest_bearing_cost_pct"]))/100
        hc=params.set_index("bucket").loc[BUCKETS,"haircut"].to_numpy(float)
        vol=float(np.std(np.diff(yc_gov[yc_gov["maturity_year"]==10]["yld_pct"].to_numpy(float)[:max(t+1,2)]))) if t>1 else 0.0
        mu=mu-hc*max(vol,0)/100

        lambdas=(cfg.lambda_var,cfg.lambda_cap,cfg.lambda_liq)
        if learner is not None:
            s=learner.state(float(hard_prob[t]),feasible_slacks(w,reg,cfg,floors,lo,hi)); a=learner.choose(s,eps=0.0); lambdas=learner.lambdas(a)
        if cfg.use_mpc and t<len(panel)-1:
            H=min(cfg.mpc_horizon,len(panel)-t)
            growth=panel["g_asset_growth_q"].iloc[t:t+H].to_numpy(float)
            mus=[]
            for kk in range(H):
                rr=panel.iloc[t+kk]; mus.append((yields.iloc[t+kk].to_numpy(float)-float(rr["rL_avg_interest_bearing_cost_pct"]))/100)
            wn,xinc,diag=solve_mpc_incremental(w,growth,mus,cov,reg,cfg,floors,lo,hi,lambdas)
        else:
            wn,xinc,diag=solve_incremental_allocation(w,float(row["g_asset_growth_q"]),mu,cov,reg,cfg,floors,lo,hi,lambdas)
        pcont=pexp=phard=0.0
        for k in range(3):
            lab=state_map[k]
            if lab=="controllable":pcont+=gamma[t,k]
            elif lab=="explicit":pexp+=gamma[t,k]
            else:phard+=gamma[t,k]
        rec={"period":row["period"],"p_controllable":pcont,"p_explicit":pexp,"p_hardening":phard,
             "regime":max([(pcont,"controllable"),(pexp,"explicit"),(phard,"hardening")])[1],
             "lambda_var_tuned":lambdas[0],"lambda_cap_tuned":lambdas[1],"lambda_liq_tuned":lambdas[2],
             "solver_status":diag["status"],"objective":diag["objective"],"growth_share":diag.get("growth_share",np.nan),
             "CAR_floor":floors["CAR"],"LCR_floor":floors["LCR"],"NSFR_floor":floors["NSFR"]}
        for i,b in enumerate(BUCKETS): rec[f"w_mpc_{b}"]=float(wn[i]); rec[f"inc_{b}"]=float(xinc[i]); rec[f"mu_{b}"]=float(mu[i])
        for k,v in diag["metrics"].items():rec[f"metric_{k}"]=float(v)
        for k,v in diag["slacks"].items():rec[f"slack_{k}"]=float(v)
        rows.append(rec); w=wn
    hist=pd.DataFrame(rows)

    # Latest-period deliverable diagnostics: bandwidth, local boundary, shadow prices.
    t=len(panel)-1; row=panel.iloc[t]; wstar=hist[[f"w_mpc_{b}" for b in BUCKETS]].iloc[-1].to_numpy(float)
    wprev=hist[[f"w_mpc_{b}" for b in BUCKETS]].iloc[-2].to_numpy(float) if len(hist)>1 else wstar.copy()
    w_obs=row[[f"w_{b}" for b in BUCKETS]].to_numpy(float); reg=RegulatoryMap.calibrate(row,w_obs,params,target_sum=float(np.sum(wstar))); floors={"CAR":float(hist.iloc[-1]["CAR_floor"]),"LCR":float(hist.iloc[-1]["LCR_floor"]),"NSFR":float(hist.iloc[-1]["NSFR_floor"])}
    bandwidth=compute_bandwidth(wstar,reg,cfg,floors,lo,hi)
    mu_latest=hist[[f"mu_{b}" for b in BUCKETS]].iloc[-1].to_numpy(float)
    shadows=approximate_shadow_prices(wstar,mu_latest,cov,wprev,reg,cfg,floors)
    eta,bindings=boundary_step(wstar,wstar-wprev,reg,cfg,floors,lo,hi)
    latest_diag={"eta_max":eta,"binding_report":bindings,"shadow_prices":shadows,"metrics":reg.metrics(wstar),
                 "slacks":feasible_slacks(wstar,reg,cfg,floors,lo,hi),"feasible":is_feasible(wstar,reg,cfg,floors,lo,hi,1e-3)}
    return {"history":hist,"bandwidth":bandwidth,"latest_diagnostics":latest_diag,"hmm":hmm,"gamma":gamma,
            "state_map":state_map,"hmm_loglik":hmm_ll,"bayes_transition":A_bayes,"bayes_nim":bayes,
            "quantile_forecast":qforecast,"bucket_yields":yields,"penalty_learner":learner}

# 附录编号统一入口
run_a6 = run_allocation_engine
