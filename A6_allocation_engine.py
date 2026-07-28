# -*- coding: utf-8 -*-
from __future__ import annotations
import math, copy, warnings, json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence
import numpy as np
import pandas as pd
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

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    RL_OK = True
except Exception:
    gym = None
    PPO = None
    RL_OK = False
from common import BUCKETS, BUCKET_CN
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
    """Map disclosed CAR/LCR/NSFR into linear portfolio constraints.

    The map is calibrated from the observed balance-sheet state.  For a
    candidate six-bucket weight vector w, the regulatory ratios are recomputed
    from implied RWA, HQLA and RSF amounts.  This lets CVXPY impose the ratios as
    true hard constraints after algebraic rearrangement.
    """
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
    def calibrate(
        cls,
        row: pd.Series,
        w_anchor: np.ndarray,
        params: pd.DataFrame,
        target_sum: float | None = None,
    ) -> "RegulatoryMap":
        pp = params.set_index("bucket").loc[BUCKETS]
        omega = pp["omega_rwa_density"].to_numpy(float)
        hqla = pp["hqla_ratio"].to_numpy(float)
        rsf = pp["rsf"].to_numpy(float)
        duration = pp["duration_year"].to_numpy(float)

        A = float(row["A_total_assets_rmb_bn"])
        target = float(np.sum(w_anchor) if target_sum is None else target_sum)
        rwa = max(A * float(omega @ w_anchor), 1e-10)
        capital_amt = float(row["CAR_pct"]) / 100.0 * rwa

        hqla_amt = max(A * float(hqla @ w_anchor), 1e-10)
        nco_amt = hqla_amt / max(float(row["LCR_pct"]) / 100.0, 1e-8)

        rsf_amt = max(A * float(rsf @ w_anchor), 1e-10)
        asf_amt = float(row["NSFR_pct"]) / 100.0 * rsf_amt

        return cls(
            A_total=A,
            capital_amt=capital_amt,
            nco_amt=nco_amt,
            asf_amt=asf_amt,
            omega=omega,
            hqla=hqla,
            rsf=rsf,
            duration=duration,
            target_sum=target,
        )

    def limits(self, floors: dict[str, float]) -> dict[str, float]:
        car = max(float(floors["CAR"]) / 100.0, 1e-8)
        lcr = max(float(floors["LCR"]) / 100.0, 1e-8)
        nsfr = max(float(floors["NSFR"]) / 100.0, 1e-8)
        return {
            # CAR >= floor  <=> omega'w <= capital / (A * CAR_floor)
            "rwa_density_max": self.capital_amt / max(self.A_total * car, 1e-12),
            # LCR >= floor  <=> hqla'w >= (NCO/A) * LCR_floor
            "hqla_ratio_min": (self.nco_amt / max(self.A_total, 1e-12)) * lcr,
            # NSFR >= floor <=> rsf'w <= (ASF/A) / NSFR_floor
            "rsf_intensity_max": (self.asf_amt / max(self.A_total, 1e-12)) / nsfr,
        }

    def metrics(self, w: np.ndarray) -> dict[str, float]:
        w = np.asarray(w, float)
        rwa = max(self.A_total * float(self.omega @ w), 1e-10)
        hqla_amt = max(self.A_total * float(self.hqla @ w), 1e-10)
        rsf_amt = max(self.A_total * float(self.rsf @ w), 1e-10)
        return {
            "CAR": 100.0 * self.capital_amt / rwa,
            "LCR": 100.0 * hqla_amt / max(self.nco_amt, 1e-10),
            "NSFR": 100.0 * self.asf_amt / rsf_amt,
            "duration": float(self.duration @ w) / max(float(np.sum(w)), 1e-10),
            "RWA_intensity": float(self.omega @ w),
            "HQLA_ratio": float(self.hqla @ w),
            "RSF_intensity": float(self.rsf @ w),
        }


def _require_cvxpy() -> None:
    if not CVXPY_OK:
        raise ImportError(
            "A6 formal engine requires cvxpy + OSQP/ECOS. Install with: "
            "pip install cvxpy osqp ecos"
        )


def _require_rl() -> None:
    if not RL_OK:
        raise ImportError(
            "A6 PPO tuner requires gymnasium + stable-baselines3 (+ torch). Install with: "
            "pip install gymnasium stable-baselines3 torch"
        )


def _as_psd(Sigma: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = np.asarray(Sigma, float)
    S = np.nan_to_num((S + S.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    eig = np.linalg.eigvalsh(S)
    if eig.min() < eps:
        S = S + np.eye(S.shape[0]) * (eps - eig.min())
    return S


def effective_constraint_floors(
    row: pd.Series,
    cfg: AllocationConfig,
    qforecast: dict[str, dict[float, float]] | None = None,
) -> dict[str, float]:
    floors = {"CAR": cfg.car_min, "LCR": cfg.lcr_min, "NSFR": cfg.nsfr_min}
    if not qforecast:
        return floors
    mapping = {"CAR": "CAR_pct", "LCR": "LCR_pct", "NSFR": "NSFR_pct"}
    for key, col in mapping.items():
        if col in qforecast and cfg.q_delta in qforecast[col]:
            qlow = float(qforecast[col][cfg.q_delta])
            current = float(row[col])
            floors[key] += max(0.0, current - qlow)
    return floors


def feasible_slacks(
    w: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
) -> dict[str, float]:
    m = reg.metrics(w)
    return {
        "CAR": m["CAR"] - floors["CAR"],
        "LCR": m["LCR"] - floors["LCR"],
        "NSFR": m["NSFR"] - floors["NSFR"],
        "duration": cfg.duration_cap - m["duration"],
        "trading": cfg.trading_cap - float(w[3]),
        "lower_min": float(np.min(np.asarray(w) - w_lower)),
        "upper_min": float(np.min(w_upper - np.asarray(w))),
        "sum_error": abs(float(np.sum(w)) - reg.target_sum),
    }


def is_feasible(
    w: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    s = feasible_slacks(w, reg, cfg, floors, w_lower, w_upper)
    hard = ["CAR", "LCR", "NSFR", "duration", "trading", "lower_min", "upper_min"]
    return min(s[k] for k in hard) >= -tol and s["sum_error"] <= max(1e-5, tol * 10)


def _cvx_constraints(
    w_expr,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    prefix: str = "",
) -> tuple[list[Any], dict[str, Any]]:
    """Create DCP-compliant hard constraints and named dual handles."""
    lim = reg.limits(floors)
    cons: list[Any] = []
    named: dict[str, Any] = {}

    def add(name: str, c) -> None:
        cons.append(c)
        named[f"{prefix}{name}"] = c

    add("SUM", cp.sum(w_expr) == reg.target_sum)
    add("LOWER", w_expr >= w_lower)
    add("UPPER", w_expr <= w_upper)
    add("CAR", reg.omega @ w_expr <= lim["rwa_density_max"])
    add("LCR", reg.hqla @ w_expr >= lim["hqla_ratio_min"])
    add("NSFR", reg.rsf @ w_expr <= lim["rsf_intensity_max"])
    add("DURATION", reg.duration @ w_expr <= cfg.duration_cap * reg.target_sum)
    add("TRADING", w_expr[3] <= cfg.trading_cap)
    return cons, named


def _solver_kwargs(solver: str, cfg: AllocationConfig) -> dict[str, Any]:
    s = solver.upper()
    if s == "OSQP":
        return {
            "max_iter": cfg.solver_maxiter,
            "eps_abs": cfg.solver_eps_abs,
            "eps_rel": cfg.solver_eps_rel,
            "polishing": True,
            "verbose": cfg.solver_verbose,
        }
    if s == "ECOS":
        return {
            "max_iters": cfg.solver_maxiter,
            "abstol": cfg.solver_eps_abs,
            "reltol": cfg.solver_eps_rel,
            "feastol": max(cfg.solver_eps_abs, 1e-8),
            "verbose": cfg.solver_verbose,
        }
    if s == "CLARABEL":
        return {"max_iter": cfg.solver_maxiter, "verbose": cfg.solver_verbose}
    if s == "SCS":
        return {"max_iters": cfg.solver_maxiter, "eps": max(cfg.solver_eps_abs, 1e-6), "verbose": cfg.solver_verbose}
    return {"verbose": cfg.solver_verbose}


def _solve_problem(problem, cfg: AllocationConfig) -> tuple[str, str]:
    _require_cvxpy()
    installed = {str(s).upper() for s in cp.installed_solvers()}
    errors = []
    candidates = list(cfg.cvxpy_solvers)
    # Numerical fallback remains inside CVXPY; the formal engine never switches
    # its primary solve to SLSQP.
    for extra in ("CLARABEL", "SCS"):
        if extra not in candidates:
            candidates.append(extra)
    for solver in candidates:
        solver = str(solver).upper()
        if solver not in installed:
            continue
        try:
            problem.solve(solver=solver, **_solver_kwargs(solver, cfg))
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return str(problem.status), solver
            errors.append(f"{solver}:{problem.status}")
        except Exception as exc:
            errors.append(f"{solver}:{type(exc).__name__}:{exc}")
    raise RuntimeError(
        "CVXPY could not solve the A6 optimisation. Installed solvers="
        f"{sorted(installed)}; attempts={errors}. Install OSQP and ECOS."
    )


def _dual_scalar(c) -> float:
    try:
        v = np.asarray(c.dual_value, float)
        if v.size == 0 or not np.all(np.isfinite(v)):
            return 0.0
        return float(np.max(v))
    except Exception:
        return 0.0


def _extract_regulatory_duals(named: dict[str, Any], prefix: str = "") -> dict[str, float]:
    return {
        "mu_CAR": _dual_scalar(named.get(f"{prefix}CAR")),
        "mu_LCR": _dual_scalar(named.get(f"{prefix}LCR")),
        "mu_NSFR": _dual_scalar(named.get(f"{prefix}NSFR")),
        "mu_DURATION": _dual_scalar(named.get(f"{prefix}DURATION")),
        "mu_TRADING": _dual_scalar(named.get(f"{prefix}TRADING")),
    }


def _utility_expr(
    w_expr,
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    lambdas: tuple[float, float, float] | None = None,
):
    """Concave utility used by both single-step optimisation and MPC."""
    lv, lc, ll = lambdas or (cfg.lambda_var, cfg.lambda_cap, cfg.lambda_liq)
    S = _as_psd(Sigma)
    lim_soft = reg.limits({
        "CAR": floors["CAR"] + cfg.safety_car_margin,
        "LCR": floors["LCR"] + cfg.safety_lcr_margin,
        "NSFR": floors["NSFR"] + cfg.safety_nsfr_margin,
    })

    expected_return = mu @ w_expr
    variance = cp.quad_form(w_expr, S)
    turnover = cp.norm1(w_expr - w_prev)

    # Capital and liquidity penalties are soft margins *inside* the hard
    # regulatory feasible set.  Their coefficients are the only quantities
    # tuned by PPO.
    cap_usage = cfg.capital_shadow_base * (reg.omega @ w_expr)
    cap_near = cp.pos((reg.omega @ w_expr) - lim_soft["rwa_density_max"])
    liq_gap = cp.pos(lim_soft["hqla_ratio_min"] - (reg.hqla @ w_expr))
    nsfr_gap = cp.pos((reg.rsf @ w_expr) - lim_soft["rsf_intensity_max"])
    liq_usage = cfg.liquidity_shadow_base * (
        cfg.kappa_lcr * liq_gap + cfg.kappa_nsfr * nsfr_gap
    )

    return (
        expected_return
        - lv * variance
        - lc * (cap_usage + cap_near)
        - ll * liq_usage
        - cfg.lambda_turn * turnover
    )


def portfolio_utility(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    lambdas: tuple[float, float, float] | None = None,
) -> float:
    """Numpy equivalent used for diagnostics and PPO rewards."""
    lv, lc, ll = lambdas or (cfg.lambda_var, cfg.lambda_cap, cfg.lambda_liq)
    w = np.asarray(w, float)
    S = _as_psd(Sigma)
    m = reg.metrics(w)
    lim_soft = reg.limits({
        "CAR": floors["CAR"] + cfg.safety_car_margin,
        "LCR": floors["LCR"] + cfg.safety_lcr_margin,
        "NSFR": floors["NSFR"] + cfg.safety_nsfr_margin,
    })
    ret = float(mu @ w)
    var = float(w @ S @ w)
    turn = float(np.sum(np.abs(w - w_prev)))
    cap_usage = cfg.capital_shadow_base * m["RWA_intensity"]
    cap_near = max(0.0, m["RWA_intensity"] - lim_soft["rwa_density_max"])
    liq_gap = max(0.0, lim_soft["hqla_ratio_min"] - m["HQLA_ratio"])
    nsfr_gap = max(0.0, m["RSF_intensity"] - lim_soft["rsf_intensity_max"])
    liq_usage = cfg.liquidity_shadow_base * (cfg.kappa_lcr * liq_gap + cfg.kappa_nsfr * nsfr_gap)
    return ret - lv * var - lc * (cap_usage + cap_near) - ll * liq_usage - cfg.lambda_turn * turn


def project_to_feasible(
    w0: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
) -> tuple[np.ndarray, str]:
    _require_cvxpy()
    w = cp.Variable(len(BUCKETS))
    constraints, _ = _cvx_constraints(w, reg, cfg, floors, w_lower, w_upper)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w - np.asarray(w0, float))), constraints)
    try:
        status, solver = _solve_problem(problem, cfg)
        out = np.asarray(w.value, float).reshape(-1)
        return out, f"{status}:{solver}"
    except Exception as exc:
        raise RuntimeError(f"No feasible A6 portfolio under current hard constraints: {exc}") from exc


def solve_weight_cvxpy(
    w_anchor: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    lambdas: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    anchor, projection_status = project_to_feasible(w_anchor, reg, cfg, floors, w_lower, w_upper)
    w = cp.Variable(len(BUCKETS))
    constraints, named = _cvx_constraints(w, reg, cfg, floors, w_lower, w_upper)
    objective = cp.Maximize(_utility_expr(w, mu, Sigma, anchor, reg, cfg, floors, lambdas))
    problem = cp.Problem(objective, constraints)
    status, solver = _solve_problem(problem, cfg)
    w_star = np.asarray(w.value, float).reshape(-1)
    return w_star, {
        "status": f"{status}:{solver}",
        "projection": projection_status,
        "objective": float(problem.value),
        "metrics": reg.metrics(w_star),
        "slacks": feasible_slacks(w_star, reg, cfg, floors, w_lower, w_upper),
        "duals": _extract_regulatory_duals(named),
    }


def _growth_share(growth_q: float) -> float:
    g = max(float(growth_q), 0.0)
    return min(max(g / (1.0 + g), 1e-4), 0.25)


def solve_incremental_allocation(
    w_prev: np.ndarray,
    growth_q: float,
    mu: np.ndarray,
    Sigma: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    lambdas: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """CVXPY stock/increment dual-track solve.

    x is the allocation mix of *new* assets.  Existing stock is inherited by
    w_t = (1-a_t) w_{t-1} + a_t x_t, so the optimiser cannot instantaneously
    rewrite the whole balance sheet.
    """
    _require_cvxpy()
    n = len(BUCKETS)
    a = _growth_share(growth_q)
    target = reg.target_sum

    x = cp.Variable(n)
    w_expr = (1.0 - a) * np.asarray(w_prev, float) + a * x
    constraints = [cp.sum(x) == target, x >= 0.0, x <= target]
    hard, named = _cvx_constraints(w_expr, reg, cfg, floors, w_lower, w_upper)
    constraints += hard
    objective = cp.Maximize(_utility_expr(w_expr, mu, Sigma, w_prev, reg, cfg, floors, lambdas))
    problem = cp.Problem(objective, constraints)

    try:
        status, solver = _solve_problem(problem, cfg)
        x_star = np.asarray(x.value, float).reshape(-1)
        w_star = (1.0 - a) * np.asarray(w_prev, float) + a * x_star
        diag = {
            "status": f"optimal_cvx_incremental:{status}:{solver}",
            "growth_share": a,
            "objective": float(problem.value),
            "metrics": reg.metrics(w_star),
            "slacks": feasible_slacks(w_star, reg, cfg, floors, w_lower, w_upper),
            "duals": _extract_regulatory_duals(named),
        }
        return w_star, x_star, diag
    except Exception:
        # A CVXPY direct-weight repair is still within the formal convex
        # optimisation stack; no SLSQP substitution is made.
        w_star, dd = solve_weight_cvxpy(
            w_prev, mu, Sigma, reg, cfg, floors, w_lower, w_upper, lambdas
        )
        x_star = (w_star - (1.0 - a) * np.asarray(w_prev, float)) / a
        # If the inherited-stock equation is numerically incompatible with a
        # nonnegative x, report the repaired stock and use a normalised display
        # mix rather than pretending the inversion is exact.
        x_show = np.maximum(x_star, 0.0)
        if x_show.sum() > 0:
            x_show *= target / x_show.sum()
        else:
            x_show = np.full(n, target / n)
        dd["status"] = "cvx_direct_weight_repair:" + dd["status"]
        dd["growth_share"] = a
        return w_star, x_show, dd


def solve_mpc_incremental(
    w_prev: np.ndarray,
    growth_forecast: Sequence[float],
    mu_forecast: Sequence[np.ndarray],
    Sigma: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
    lambdas: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Multi-period convex MPC; returns only the first control action."""
    _require_cvxpy()
    H = min(cfg.mpc_horizon, len(growth_forecast), len(mu_forecast))
    if H <= 1:
        return solve_incremental_allocation(
            w_prev, growth_forecast[0], mu_forecast[0], Sigma, reg, cfg,
            floors, w_lower, w_upper, lambdas
        )

    n = len(BUCKETS)
    target = reg.target_sum
    X = cp.Variable((H, n))
    constraints: list[Any] = []
    named_first: dict[str, Any] = {}
    W: list[Any] = []
    prev_expr: Any = np.asarray(w_prev, float)

    for k in range(H):
        a = _growth_share(float(growth_forecast[k]))
        wk = (1.0 - a) * prev_expr + a * X[k, :]
        W.append(wk)
        constraints += [cp.sum(X[k, :]) == target, X[k, :] >= 0.0, X[k, :] <= target]
        hard, named = _cvx_constraints(wk, reg, cfg, floors, w_lower, w_upper, prefix=f"t{k}_")
        constraints += hard
        if k == 0:
            named_first = named
        prev_expr = wk

    value = 0
    prev_for_turn: Any = np.asarray(w_prev, float)
    for k in range(H):
        value += (cfg.discount ** k) * _utility_expr(
            W[k], np.asarray(mu_forecast[k], float), Sigma, prev_for_turn,
            reg, cfg, floors, lambdas
        )
        prev_for_turn = W[k]

    problem = cp.Problem(cp.Maximize(value), constraints)
    try:
        status, solver = _solve_problem(problem, cfg)
        x0 = np.asarray(X.value[0], float).reshape(-1)
        a0 = _growth_share(float(growth_forecast[0]))
        w0 = (1.0 - a0) * np.asarray(w_prev, float) + a0 * x0
        return w0, x0, {
            "status": f"optimal_cvx_mpc:{status}:{solver}",
            "growth_share": a0,
            "objective": float(problem.value),
            "metrics": reg.metrics(w0),
            "slacks": feasible_slacks(w0, reg, cfg, floors, w_lower, w_upper),
            "duals": _extract_regulatory_duals(named_first, prefix="t0_"),
            "mpc_horizon_used": H,
        }
    except Exception:
        return solve_incremental_allocation(
            w_prev, growth_forecast[0], mu_forecast[0], Sigma, reg, cfg,
            floors, w_lower, w_upper, lambdas
        )


def compute_bandwidth(
    w_anchor: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
) -> pd.DataFrame:
    """CVXPY feasible-set scan for per-bucket lower/upper configuration bands."""
    _require_cvxpy()
    anchor, _ = project_to_feasible(w_anchor, reg, cfg, floors, w_lower, w_upper)
    rows = []
    for i, b in enumerate(BUCKETS):
        # Lower boundary
        w_lo = cp.Variable(len(BUCKETS))
        cons_lo, _ = _cvx_constraints(w_lo, reg, cfg, floors, w_lower, w_upper)
        p_lo = cp.Problem(
            cp.Minimize(w_lo[i] + cfg.bandwidth_rho * cp.sum_squares(w_lo - anchor)),
            cons_lo,
        )
        try:
            st_lo, sol_lo = _solve_problem(p_lo, cfg)
            lower = float(w_lo.value[i])
            low_status = f"{st_lo}:{sol_lo}"
        except Exception as exc:
            lower = float(anchor[i])
            low_status = f"failed:{type(exc).__name__}"

        # Upper boundary
        w_hi = cp.Variable(len(BUCKETS))
        cons_hi, _ = _cvx_constraints(w_hi, reg, cfg, floors, w_lower, w_upper)
        p_hi = cp.Problem(
            cp.Maximize(w_hi[i] - cfg.bandwidth_rho * cp.sum_squares(w_hi - anchor)),
            cons_hi,
        )
        try:
            st_hi, sol_hi = _solve_problem(p_hi, cfg)
            upper = float(w_hi.value[i])
            high_status = f"{st_hi}:{sol_hi}"
        except Exception as exc:
            upper = float(anchor[i])
            high_status = f"failed:{type(exc).__name__}"

        rows.append({
            "bucket": b,
            "bucket_cn": BUCKET_CN[b],
            "anchor": float(anchor[i]),
            "lower": lower,
            "upper": upper,
            "width": max(0.0, upper - lower),
            "lower_status": low_status,
            "upper_status": high_status,
        })
    return pd.DataFrame(rows)


def boundary_step(
    w_anchor: np.ndarray,
    direction: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    w_lower: np.ndarray,
    w_upper: np.ndarray,
) -> tuple[float, list[str]]:
    """One-dimensional boundary scan used only for reporting η_max."""
    d = np.asarray(direction, float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return 0.0, []
    d = d / norm

    def ok(eta: float) -> bool:
        return is_feasible(w_anchor + eta * d, reg, cfg, floors, w_lower, w_upper, 1e-7)

    lo, hi = 0.0, 0.01
    while hi < 2.0 and ok(hi):
        lo, hi = hi, hi * 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if ok(mid):
            lo = mid
        else:
            hi = mid

    w_edge = w_anchor + lo * d
    s = feasible_slacks(w_edge, reg, cfg, floors, w_lower, w_upper)
    tolerances = {
        "CAR": 1e-3,
        "LCR": 1e-2,
        "NSFR": 1e-2,
        "duration": 1e-3,
        "trading": 1e-4,
        "lower_min": 1e-4,
        "upper_min": 1e-4,
    }
    binding = [k for k, tol in tolerances.items() if s[k] <= tol]
    return float(lo), binding


def _rl_observation(
    row: pd.Series,
    w: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    hard_prob: float,
    w_lower: np.ndarray,
    w_upper: np.ndarray,
) -> np.ndarray:
    sl = feasible_slacks(w, reg, cfg, floors, w_lower, w_upper)
    target = max(reg.target_sum, 1e-8)
    obs = np.r_[
        np.clip(hard_prob, 0.0, 1.0),
        np.clip(sl["CAR"] / 5.0, -5.0, 5.0),
        np.clip(sl["LCR"] / 50.0, -5.0, 5.0),
        np.clip(sl["NSFR"] / 10.0, -5.0, 5.0),
        np.clip(float(row["NIM_state_pct"]) / 3.0, -5.0, 5.0),
        np.clip(float(row["LPR1_pct"]) / 5.0, -5.0, 5.0),
        np.clip(float(row["DR007_pct"]) / 4.0, -5.0, 5.0),
        np.clip(float(row["g_asset_growth_q"]) * 10.0, -5.0, 5.0),
        np.asarray(w, float) / target,
    ].astype(np.float32)
    return np.clip(obs, -10.0, 10.0)


def _action_to_lambdas(action: np.ndarray, cfg: AllocationConfig) -> tuple[float, float, float]:
    a = np.clip(np.asarray(action, float).reshape(3), -1.0, 1.0)
    z = (a + 1.0) / 2.0
    lo = max(cfg.ppo_lambda_min_mult, 1e-4)
    hi = max(cfg.ppo_lambda_max_mult, lo + 1e-4)
    mult = np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo)))
    base = np.array([cfg.lambda_var, cfg.lambda_cap, cfg.lambda_liq], float)
    return tuple((base * mult).tolist())


if RL_OK:
    class PenaltyTuneEnv(gym.Env):
        """Governance-safe PPO environment.

        PPO observes regime/constraint/balance-sheet state and chooses only
        three penalty multipliers.  It never has an action dimension for any
        asset weight.  The six asset weights are always produced by CVXPY.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            panel: pd.DataFrame,
            params: pd.DataFrame,
            yields: pd.DataFrame,
            hard_prob: np.ndarray,
            Sigma: np.ndarray,
            cfg: AllocationConfig,
        ):
            super().__init__()
            self.panel = panel.reset_index(drop=True)
            self.params = params
            self.yields = yields.reset_index(drop=True)
            self.hard_prob = np.asarray(hard_prob, float)
            self.Sigma = _as_psd(Sigma)
            self.cfg = cfg
            self.lower = np.asarray(cfg.w_lower, float)
            self.upper = np.asarray(cfg.w_upper, float)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(8 + len(BUCKETS),), dtype=np.float32)
            self.t = 0
            self.w = self.panel[[f"w_{b}" for b in BUCKETS]].iloc[0].to_numpy(float)

        def _state(self) -> np.ndarray:
            row = self.panel.iloc[self.t]
            w_obs = row[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
            reg = RegulatoryMap.calibrate(row, w_obs, self.params, target_sum=float(np.sum(self.w)))
            floors = {"CAR": self.cfg.car_min, "LCR": self.cfg.lcr_min, "NSFR": self.cfg.nsfr_min}
            return _rl_observation(
                row, self.w, reg, self.cfg, floors, float(self.hard_prob[self.t]),
                self.lower, self.upper
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            self.w = self.panel[[f"w_{b}" for b in BUCKETS]].iloc[0].to_numpy(float)
            return self._state(), {}

        def step(self, action):
            row = self.panel.iloc[self.t]
            w_obs = row[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
            reg = RegulatoryMap.calibrate(row, w_obs, self.params, target_sum=float(np.sum(self.w)))
            floors = {"CAR": self.cfg.car_min, "LCR": self.cfg.lcr_min, "NSFR": self.cfg.nsfr_min}
            mu = (
                self.yields.iloc[self.t].to_numpy(float)
                - float(row["rL_avg_interest_bearing_cost_pct"])
            ) / 100.0
            lambdas = _action_to_lambdas(action, self.cfg)

            solver_failed = False
            try:
                w_new, _, diag = solve_incremental_allocation(
                    self.w,
                    float(row["g_asset_growth_q"]),
                    mu,
                    self.Sigma,
                    reg,
                    self.cfg,
                    floors,
                    self.lower,
                    self.upper,
                    lambdas,
                )
            except Exception:
                w_new = self.w.copy()
                solver_failed = True
                diag = {
                    "objective": -1.0,
                    "metrics": reg.metrics(w_new),
                    "slacks": feasible_slacks(w_new, reg, self.cfg, floors, self.lower, self.upper),
                    "status": "rl_solver_failure",
                }

            violation = sum(
                max(0.0, -float(diag["slacks"].get(k, 0.0)))
                for k in ("CAR", "LCR", "NSFR", "duration", "trading", "lower_min", "upper_min")
            )
            reward = 100.0 * float(diag.get("objective", -1.0)) - 200.0 * violation
            if solver_failed:
                reward -= 100.0

            self.w = np.asarray(w_new, float)
            self.t += 1
            terminated = self.t >= len(self.panel) - 1
            truncated = False
            if terminated:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                obs = self._state()
            info = {
                "lambda_var": lambdas[0],
                "lambda_cap": lambdas[1],
                "lambda_liq": lambdas[2],
                "solver_status": diag.get("status", ""),
            }
            return obs, float(reward), terminated, truncated, info
else:
    class PenaltyTuneEnv:  # pragma: no cover - dependency error stub
        def __init__(self, *args, **kwargs):
            _require_rl()


def train_penalty_ppo(
    panel: pd.DataFrame,
    params: pd.DataFrame,
    yields: pd.DataFrame,
    hard_prob: np.ndarray,
    Sigma: np.ndarray,
    cfg: AllocationConfig,
):
    _require_cvxpy()
    _require_rl()
    env = PenaltyTuneEnv(panel, params, yields, hard_prob, Sigma, cfg)
    n_steps = int(max(8, min(cfg.ppo_n_steps, max(8, len(panel) - 1))))
    batch_size = int(max(8, min(cfg.ppo_batch_size, n_steps)))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.ppo_learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=cfg.ppo_gamma,
        gae_lambda=cfg.ppo_gae_lambda,
        clip_range=cfg.ppo_clip_range,
        ent_coef=cfg.ppo_ent_coef,
        seed=cfg.seed,
        verbose=cfg.ppo_verbose,
        device=cfg.ppo_device,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=int(cfg.ppo_timesteps), progress_bar=False)
    return model


def _ppo_lambdas_for_state(
    model,
    row: pd.Series,
    w: np.ndarray,
    reg: RegulatoryMap,
    cfg: AllocationConfig,
    floors: dict[str, float],
    hard_prob: float,
    w_lower: np.ndarray,
    w_upper: np.ndarray,
) -> tuple[float, float, float]:
    obs = _rl_observation(row, w, reg, cfg, floors, hard_prob, w_lower, w_upper)
    action, _ = model.predict(obs, deterministic=True)
    return _action_to_lambdas(action, cfg)


def run_allocation_engine(
    panel: pd.DataFrame,
    yc_gov: pd.DataFrame,
    yc_cred: pd.DataFrame,
    params: pd.DataFrame,
    cfg: AllocationConfig,
) -> dict[str, Any]:
    """A6 complete formal engine: HMM + Bayesian drift + quantiles + PPO + CVXPY MPC."""
    _require_cvxpy()
    if cfg.use_rl:
        _require_rl()

    # -------------------------
    # 1) Regime learning (HMM)
    # -------------------------
    feats, fnames = build_hmm_features(panel, params)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feats)
    hmm = init_hmm(Xs, 3, cfg.seed)
    hmm_ll = hmm.em_train(Xs, 50, 1e-4)
    gamma, xi, _ = hmm.forward_backward(Xs)
    state_map = map_hmm_regimes(hmm, scaler, fnames)
    A_bayes = bayes_transition_mean(xi, 1.0)
    hard_idx = [k for k, v in state_map.items() if v == "hardening"]
    hard_prob = gamma[:, hard_idx].sum(axis=1) if hard_idx else np.zeros(len(panel))

    # --------------------------------------
    # 2) Bayesian/Kalman parameter drift
    # --------------------------------------
    bayes = dynamic_bayesian_nim_regression(panel, hard_prob) if cfg.use_bayes else None

    # --------------------------------------
    # 3) Quantile constraint forecasts
    # --------------------------------------
    qforecast = (
        quantile_forecast_latest(
            panel, yc_gov, gamma,
            (cfg.q_delta, 0.5, 1.0 - cfg.q_delta),
            cfg.seed,
        )
        if cfg.use_quantile else {}
    )
    yields = derive_bucket_yields(panel, yc_gov, yc_cred)

    # --------------------------------------
    # 4) Risk covariance for robust utility
    # --------------------------------------
    Yret = yields.to_numpy(float) / 100.0
    if len(Yret) > 8:
        cov = np.cov(np.diff(Yret, axis=0).T)
    else:
        cov = np.diag(np.full(len(BUCKETS), 1e-4))
    cov = _as_psd(cov)
    lower = np.asarray(cfg.w_lower, float)
    upper = np.asarray(cfg.w_upper, float)

    # --------------------------------------
    # 5) Governance-safe PPO penalty tuner
    # --------------------------------------
    ppo_model = None
    if cfg.use_rl:
        ppo_model = train_penalty_ppo(panel, params, yields, hard_prob, cov, cfg)

    # --------------------------------------
    # 6) Rolling stock/increment + MPC solve
    # --------------------------------------
    rows = []
    w = panel[[f"w_{b}" for b in BUCKETS]].iloc[0].to_numpy(float)

    for t in range(len(panel)):
        row = panel.iloc[t]
        w_obs = row[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
        reg = RegulatoryMap.calibrate(row, w_obs, params, target_sum=float(np.sum(w)))

        qf = qforecast if t == len(panel) - 1 else None
        floors = effective_constraint_floors(row, cfg, qf)
        # If the uncertainty buffer itself makes the feasible set empty, fall
        # back only to the statutory/management hard floors, not to a different
        # optimisation method.
        try:
            test_anchor, _ = project_to_feasible(w, reg, cfg, floors, lower, upper)
            if not is_feasible(test_anchor, reg, cfg, floors, lower, upper, 1e-4):
                raise RuntimeError("chance-buffer infeasible")
        except Exception:
            floors = {"CAR": cfg.car_min, "LCR": cfg.lcr_min, "NSFR": cfg.nsfr_min}
            test_anchor, _ = project_to_feasible(w, reg, cfg, floors, lower, upper)
            if not is_feasible(test_anchor, reg, cfg, floors, lower, upper, 1e-4):
                raise RuntimeError(f"A6 hard feasible set is empty at period {row['period']}")

        mu = (
            yields.iloc[t].to_numpy(float)
            - float(row["rL_avg_interest_bearing_cost_pct"])
        ) / 100.0
        haircut = params.set_index("bucket").loc[BUCKETS, "haircut"].to_numpy(float)
        y10 = yc_gov[yc_gov["maturity_year"] == 10]["yld_pct"].to_numpy(float)
        vol = float(np.std(np.diff(y10[: max(t + 1, 2)]))) if t > 1 else 0.0
        mu = mu - haircut * max(vol, 0.0) / 100.0

        lambdas = (cfg.lambda_var, cfg.lambda_cap, cfg.lambda_liq)
        if ppo_model is not None:
            lambdas = _ppo_lambdas_for_state(
                ppo_model, row, w, reg, cfg, floors, float(hard_prob[t]), lower, upper
            )

        if cfg.use_mpc and t < len(panel) - 1:
            H = min(cfg.mpc_horizon, len(panel) - t)
            growth = panel["g_asset_growth_q"].iloc[t:t + H].to_numpy(float)
            mus = []
            for kk in range(H):
                rr = panel.iloc[t + kk]
                mui = (
                    yields.iloc[t + kk].to_numpy(float)
                    - float(rr["rL_avg_interest_bearing_cost_pct"])
                ) / 100.0
                mus.append(mui)
            w_new, x_inc, diag = solve_mpc_incremental(
                w, growth, mus, cov, reg, cfg, floors, lower, upper, lambdas
            )
        else:
            w_new, x_inc, diag = solve_incremental_allocation(
                w, float(row["g_asset_growth_q"]), mu, cov, reg, cfg,
                floors, lower, upper, lambdas
            )

        p_controllable = p_explicit = p_hardening = 0.0
        for k in range(3):
            label = state_map[k]
            if label == "controllable":
                p_controllable += gamma[t, k]
            elif label == "explicit":
                p_explicit += gamma[t, k]
            else:
                p_hardening += gamma[t, k]

        rec = {
            "period": row["period"],
            "p_controllable": p_controllable,
            "p_explicit": p_explicit,
            "p_hardening": p_hardening,
            "regime": max([
                (p_controllable, "controllable"),
                (p_explicit, "explicit"),
                (p_hardening, "hardening"),
            ])[1],
            "lambda_var_tuned": lambdas[0],
            "lambda_cap_tuned": lambdas[1],
            "lambda_liq_tuned": lambdas[2],
            "solver_status": diag["status"],
            "objective": diag["objective"],
            "growth_share": diag.get("growth_share", np.nan),
            "CAR_floor": floors["CAR"],
            "LCR_floor": floors["LCR"],
            "NSFR_floor": floors["NSFR"],
        }
        for i, b in enumerate(BUCKETS):
            rec[f"w_mpc_{b}"] = float(w_new[i])
            rec[f"inc_{b}"] = float(x_inc[i])
            rec[f"mu_{b}"] = float(mu[i])
        for k, v in diag["metrics"].items():
            rec[f"metric_{k}"] = float(v)
        for k, v in diag["slacks"].items():
            rec[f"slack_{k}"] = float(v)
        for k, v in diag.get("duals", {}).items():
            rec[f"dual_{k.replace('mu_', '')}"] = float(v)
        rows.append(rec)
        w = np.asarray(w_new, float)

    hist = pd.DataFrame(rows)

    # --------------------------------------
    # 7) Bandwidth, boundary and true duals
    # --------------------------------------
    t = len(panel) - 1
    row = panel.iloc[t]
    w_star = hist[[f"w_mpc_{b}" for b in BUCKETS]].iloc[-1].to_numpy(float)
    w_prev = (
        hist[[f"w_mpc_{b}" for b in BUCKETS]].iloc[-2].to_numpy(float)
        if len(hist) > 1 else w_star.copy()
    )
    w_obs = row[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
    reg = RegulatoryMap.calibrate(row, w_obs, params, target_sum=float(np.sum(w_star)))
    floors = {
        "CAR": float(hist.iloc[-1]["CAR_floor"]),
        "LCR": float(hist.iloc[-1]["LCR_floor"]),
        "NSFR": float(hist.iloc[-1]["NSFR_floor"]),
    }
    bandwidth = compute_bandwidth(w_star, reg, cfg, floors, lower, upper)
    eta, bindings = boundary_step(w_star, w_star - w_prev, reg, cfg, floors, lower, upper)
    shadow_prices = {
        "mu_CAR": float(hist.iloc[-1].get("dual_CAR", 0.0)),
        "mu_LCR": float(hist.iloc[-1].get("dual_LCR", 0.0)),
        "mu_NSFR": float(hist.iloc[-1].get("dual_NSFR", 0.0)),
        "mu_DURATION": float(hist.iloc[-1].get("dual_DURATION", 0.0)),
        "mu_TRADING": float(hist.iloc[-1].get("dual_TRADING", 0.0)),
        "source": "CVXPY_KKT_dual_value",
    }
    latest_diag = {
        "eta_max": eta,
        "binding_report": bindings,
        "shadow_prices": shadow_prices,
        "metrics": reg.metrics(w_star),
        "slacks": feasible_slacks(w_star, reg, cfg, floors, lower, upper),
        "feasible": is_feasible(w_star, reg, cfg, floors, lower, upper, 1e-4),
        "mpc_enabled": bool(cfg.use_mpc),
        "ppo_enabled": bool(cfg.use_rl),
        "optimizer": "CVXPY",
        "preferred_solvers": list(cfg.cvxpy_solvers),
    }

    return {
        "history": hist,
        "bandwidth": bandwidth,
        "latest_diagnostics": latest_diag,
        "hmm": hmm,
        "gamma": gamma,
        "state_map": state_map,
        "hmm_loglik": hmm_ll,
        "bayes_transition": A_bayes,
        "bayes_nim": bayes,
        "quantile_forecast": qforecast,
        "bucket_yields": yields,
        "ppo_model": ppo_model,
        "penalty_learner": ppo_model,  # compatibility alias
    }


# Appendix-numbered public entry point
run_a6 = run_allocation_engine
