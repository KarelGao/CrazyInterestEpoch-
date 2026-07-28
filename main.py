# -*- coding: utf-8 -*-
"""A1-A10 modular system runner.

Execution order is deliberately identical to the appendix numbering:
A1 -> A2 -> A3 -> A4 -> A5 -> A6 -> A7 -> A8 -> A9 -> A10.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

from common import BUCKETS, BUCKET_CN, _write_json, build_stage_t_input
from config import SystemConfig
from data_generator import simulate_nim_state_data
from A1_threshold_regression import run_a1
from A2_granger_causality import run_a2
from A3_shadow_rate import run_a3
from A4_local_projection import run_a4
from A5_profitability_elasticity import run_a5
from A6_allocation_engine import run_a6
from A7_rate_stage_recognition import run_a7
from A8_RORAC_RORWA import run_a8
from A9_stock_new_asset_gates import run_a9
from A10_backtest_rolling_calibration import run_a10
from outputs import _plot_outputs


def load_or_generate(cfg: SystemConfig, force_generate: bool = True):
    sim = Path(cfg.sim_dir)
    req = [sim / "panel_quarterly.csv", sim / "yc_gov_long.csv",
           sim / "yc_cred_long.csv", sim / "bucket_params.csv"]
    if force_generate or not all(p.exists() for p in req):
        simulate_nim_state_data(
            start=cfg.start, end=cfg.end, seed=cfg.seed, out_dir=cfg.sim_dir,
            leave_blank_other_assets=True, blank_share_range=(0.03, 0.08),
            lpr_scenario="stage", lpr_stage_levels=(3.0, 2.5, 2.0, 1.5),
            lpr_stage_len=8,
        )
    return tuple(pd.read_csv(p) for p in req)


def make_a5_inputs(panel: pd.DataFrame, params: pd.DataFrame):
    r = panel.iloc[-1]
    A = float(r["A_total_assets_rmb_bn"])
    w = r[[f"w_{b}" for b in BUCKETS]].to_numpy(float)
    pp = params.set_index("bucket").loc[BUCKETS]
    avgEA = A * max(w.sum(), 0.85)
    nii = float(r["NIM_state_pct"]) / 100 * avgEA
    equity = A * 0.10
    profit = max(nii + float(r.get("non_interest_income_proxy_rmb_bn", 0))
                 - float(r.get("provision_proxy_rmb_bn", 0))
                 - float(r.get("operating_cost_proxy_rmb_bn", 0)), 1e-6)
    bs = {"TotalAssets": A, "TotalLiab": A-equity, "Equity": equity, "NII": nii,
          "NIM": float(r["NIM_state_pct"])/100, "ROA": profit/A, "ROE": profit/equity,
          "AvgEarningAssets": avgEA, "Profit0": profit, "PayoutRatio": 0.30}
    ba = []
    for i, b in enumerate(BUCKETS):
        dur = float(pp.iloc[i]["duration_year"])
        freq = 4.0 if b in {"credit", "ib"} else max(1.0, 1/max(dur, 0.25))
        ba.append({"Name": b, "Balance": A*w[i], "RepriceFreq": freq,
                   "RemainMat": max(dur, 0.25), "PrepayRate": 0.02 if b == "credit" else 0.0})
    L = A-equity
    cost = float(r["rL_avg_interest_bearing_cost_pct"])/100
    bl = [
        {"Name":"deposit","Balance":L*0.72,"RepriceFreq":1.0,"RemainMat":1.0,"PrepayRate":0.0,"CostRate0":cost*0.75,"CostFloor":0.006},
        {"Name":"market_funding","Balance":L*0.18,"RepriceFreq":4.0,"RemainMat":0.5,"PrepayRate":0.0,"CostRate0":cost*1.15,"CostFloor":0.002},
        {"Name":"other_liability","Balance":L*0.10,"RepriceFreq":2.0,"RemainMat":1.0,"PrepayRate":0.0,"CostRate0":cost,"CostFloor":0.003},
    ]
    return bs, ba, bl


def run_system(cfg: SystemConfig, force_generate: bool = True):
    np.random.seed(cfg.seed)
    t0 = time.time()
    outdir = Path(cfg.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    panel, yc_gov, yc_cred, params = load_or_generate(cfg, force_generate)

    # stage_t is an INPUT of A2/A4 in the appendix, so prepare it outside A1-A10.
    stage_t = build_stage_t_input(panel)

    print("[A1] 门槛回归模型及显著性检验")
    X = np.column_stack([np.ones(len(panel)), panel["DR007_pct"], panel["g_asset_growth_q"]])
    a1 = run_a1(panel["NIM_state_pct"], X, panel["LPR1_pct"], cfg.threshold)
    a1["rssGrid"].to_csv(outdir/"A1_threshold_grid.csv", index=False, encoding="utf-8-sig")
    _write_json(outdir/"A1_threshold_summary.json", {k:v for k,v in a1.items() if k not in {"rssGrid","betaHat"}} | {"betaHat":a1["betaHat"]})

    print("[A2] Granger 因果关系检验")
    a2 = run_a2(panel["LPR1_pct"], panel["NIM_state_pct"], stage_t,
                p_max=6, ic_type="BIC", transform_rule="level", allow_stage_lag=False)
    a2.to_csv(outdir/"A2_granger_full_vs_stages.csv", index=False, encoding="utf-8-sig")

    print("[A3] Wu-Xia 影子利率估计")
    piv = yc_gov.pivot(index="period", columns="maturity_year", values="yld_pct").reindex(panel["period"].astype(str))
    mats = np.array(piv.columns, float); Y = piv.to_numpy(float)
    a3 = run_a3(Y, mats, lb=0.0, est_mode="FixParam" if cfg.fast_mode else "MLE",
                missing_rule="Interpolate", smooth=True, short_maturity=float(np.min(mats)),
                compare_series={"DR007":panel["DR007_pct"], "LPR1Y":panel["LPR1_pct"]})
    pd.DataFrame({"period":panel["period"],"rShadow":a3["rShadow"],"rNominal":a3["rNominal"]}).to_csv(outdir/"A3_shadow_rate.csv",index=False,encoding="utf-8-sig")
    a3["FitDiag"].to_csv(outdir/"A3_fit_diagnostics.csv", index=False, encoding="utf-8-sig")
    _write_json(outdir/"A3_compare_diagnostics.json", a3["CompareDiag"])

    print("[A4] 分阶段 Local Projection 动态响应")
    a4 = run_a4(panel["LPR1_pct"], panel["NIM_state_pct"], stage_t,
                hmax=6 if cfg.fast_mode else 10, p_ctrl=2, shock_scale=0.01)
    a4.to_csv(outdir/"A4_stage_local_projection.csv", index=False, encoding="utf-8-sig")

    print("[A5] 盈利弹性测算")
    bs, ba, bl = make_a5_inputs(panel, params)
    a5 = run_a5([-0.001,-0.002], bs, ba, bl, horizon_years=1.0, tax_rate=0.25)
    a5["ScenarioTable"].to_csv(outdir/"A5_profitability_elasticity.csv",index=False,encoding="utf-8-sig")
    a5["PlotData"].to_csv(outdir/"A5_plot_data.csv",index=False,encoding="utf-8-sig")
    _write_json(outdir/"A5_bucket_details.json", a5["DetailTable"])

    print("[A6] NIM-State Robust 核心资产配置")
    a6 = run_a6(panel, yc_gov, yc_cred, params, cfg.allocation)
    # Formal A6 deliverables. Keep compatibility aliases for older drafts.
    a6["history"].to_csv(outdir/"A6_allocation_path.csv",index=False,encoding="utf-8-sig")
    a6["history"].to_csv(outdir/"A6_allocation_history.csv",index=False,encoding="utf-8-sig")
    a6["bandwidth"].to_csv(outdir/"A6_bandwidth_latest.csv",index=False,encoding="utf-8-sig")
    a6["bandwidth"].to_csv(outdir/"A6_latest_bandwidth.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(a6["bayes_transition"]).to_csv(outdir/"A6_hmm_transition.csv",index=False,encoding="utf-8-sig")
    _write_json(outdir/"A6_shadow_prices_latest.json", a6["latest_diagnostics"]["shadow_prices"])
    _write_json(outdir/"A6_latest_diagnostics.json", a6["latest_diagnostics"])
    _write_json(outdir/"A6_model_diagnostics.json", {"hmm_loglik":a6["hmm_loglik"],"state_map":a6["state_map"],"bayes_transition":a6["bayes_transition"],"quantile_forecast":a6["quantile_forecast"]})
    if a6.get("ppo_model") is not None:
        a6["ppo_model"].save(str(outdir/"A6_ppo_penalty_tuner"))

    print("[A7] 外部利率阶段识别 + NIM 内部校准")
    a7 = run_a7(panel, yc_gov, cfg.stage)
    a7.to_csv(outdir/"A7_rate_stage_and_nim_calibration.csv",index=False,encoding="utf-8-sig")

    print("[A8] RORAC / RORWA 风险调整资本效率诊断")
    a8 = run_a8(panel, params, a6["bucket_yields"], a6["history"])
    a8.to_csv(outdir/"A8_RORAC_RORWA_diagnostics.csv",index=False,encoding="utf-8-sig")

    print("[A9] 存量资产分层 + 新增资源五道准入闸门")
    a9 = run_a9(a8, params)
    a9.to_csv(outdir/"A9_stock_pool_and_gates.csv",index=False,encoding="utf-8-sig")

    print("[A10] 回测、触发与滚动校准")
    a10 = run_a10(panel, a6["history"], a7, params, cfg.trigger_set)
    a10["BacktestTable"].to_csv(outdir/"A10_backtest.csv",index=False,encoding="utf-8-sig")
    a10["TriggerTable"].to_csv(outdir/"A10_trigger_and_correction.csv",index=False,encoding="utf-8-sig")

    latest=panel.iloc[-1]; h=a6["history"].iloc[-1]
    overview=pd.DataFrame([{"期别":latest["period"],"1Y LPR(%)":latest["LPR1_pct"],"DR007(%)":latest["DR007_pct"],"NIM(%)":latest["NIM_state_pct"],
                            "CAR(%)":latest["CAR_pct"],"LCR(%)":latest["LCR_pct"],"NSFR(%)":latest["NSFR_pct"],"利率阶段":a7.iloc[-1]["RateStage"],
                            "低利率阶段":a7.iloc[-1]["LowRatePhase"],"约束状态":h["regime"],"触发级别":a10["LatestTriggerLevel"]}])
    overview.to_csv(outdir/"table_overview_cn.csv",index=False,encoding="utf-8-sig")
    prev=a6["history"].iloc[-2] if len(a6["history"])>1 else h
    direction=[]
    for b in BUCKETS:
        direction.append({"资产类别":BUCKET_CN[b],"本期权重":h[f"w_mpc_{b}"],"上期权重":prev[f"w_mpc_{b}"],"变动Δ":h[f"w_mpc_{b}"]-prev[f"w_mpc_{b}"],"新增配置结构":h[f"inc_{b}"]})
    pd.DataFrame(direction).to_csv(outdir/"table_direction_cn.csv",index=False,encoding="utf-8-sig")
    _plot_outputs(outdir,panel,a7,a3,a4,a6)

    summary={"runtime_seconds":time.time()-t0,"rows":len(panel),
             "A1":{"gammaHat":a1["gammaHat"],"CIgamma":a1["CIgamma"],"SupFobs":a1["SupFobs"],"pValue":a1["pValue"],"Decision1":a1["Decision1"],"Decision2":a1["Decision2"]},
             "A3_compare":a3["CompareDiag"],"A6_latest":a6["latest_diagnostics"],"A7_latest":a7.iloc[-1].to_dict(),
             "A10_latest":{"TriggerLevel":a10["LatestTriggerLevel"],"CorrectionAction":a10["LatestCorrectionAction"],"ParameterUpdate":a10["LatestParameterUpdate"]},
             "output_dir":str(outdir.resolve())}
    _write_json(outdir/"system_summary.json",summary)
    return {"panel":panel,"A1":a1,"A2":a2,"A3":a3,"A4":a4,"A5":a5,"A6":a6,"A7":a7,"A8":a8,"A9":a9,"A10":a10,"summary":summary}


def parse_args():
    p=argparse.ArgumentParser(description="A1-A10 formal NIM-State Robust Allocation System")
    p.add_argument("--out-dir",default="./system_outputs")
    p.add_argument("--sim-dir",default="./sim_data")
    p.add_argument("--full",action="store_true",help="research-grade settings: larger bootstrap and PPO training")
    p.add_argument("--bootstrap",type=int,default=None)
    p.add_argument("--ppo-timesteps",type=int,default=None)
    p.add_argument("--no-mpc",action="store_true",help="ablation only: disable multi-period MPC and use one-step CVXPY solve")
    p.add_argument("--no-rl",action="store_true",help="ablation only: disable PPO penalty tuning")
    p.add_argument("--reuse-data",action="store_true")
    return p.parse_args()


def main():
    args=parse_args()
    cfg=SystemConfig(out_dir=args.out_dir,sim_dir=args.sim_dir,fast_mode=not args.full)
    if args.bootstrap is not None:
        cfg.threshold.bootstrap_B=max(19,args.bootstrap)
    elif args.full:
        cfg.threshold.bootstrap_B=999
    cfg.allocation.use_mpc=not bool(args.no_mpc)
    cfg.allocation.use_rl=not bool(args.no_rl)
    if args.ppo_timesteps is not None:
        cfg.allocation.ppo_timesteps=max(64,args.ppo_timesteps)
    elif args.full:
        cfg.allocation.ppo_timesteps=10000
    print("[SYSTEM] 按附录顺序运行：A1 → A2 → A3 → A4 → A5 → A6 → A7 → A8 → A9 → A10")
    result=run_system(cfg,force_generate=not args.reuse_data)
    print("[SYSTEM] 完成：",Path(cfg.out_dir).resolve())
    print("[A1]",result["A1"]["Decision1"],"p=",round(result["A1"]["pValue"],4),"gamma=",round(result["A1"]["gammaHat"],4))
    print("[A6] feasible=",result["A6"]["latest_diagnostics"]["feasible"])
    print("[A10] trigger=",result["A10"]["LatestTriggerLevel"])


if __name__ == "__main__":
    main()
