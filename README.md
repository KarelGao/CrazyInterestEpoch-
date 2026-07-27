# CrazyInterestEpoch-
本项目构建了一套面向商业银行低利率环境的大类资产配置决策系统。系统整合门槛回归、Granger 因果检验、状态空间模型、局部投影、盈利弹性测算、隐马尔可夫模型（HMM）、机器学习预测、模型预测控制（MPC）与强化学习（RL），实现“利率状态识别—盈利压力测算—六类资产配置优化—资本与流动性约束校验—存量/增量资产准入—回测与动态纠偏”的完整闭环。模型以 CAR、LCR、NSFR、RWA、久期等监管与风险指标约束可行域，输出资产配置区间、增量调整方向、约束边界与滚动优化结果，可用于银行资产负债管理与大类资产配置研究。

# NIM-State Robust Allocation Research System（A1—A10）

本程序将《伪代码.docx》中的 A1—A10 算法整合为一个可直接运行的 Python 系统，并复用、重构原有 `main.py`、`main02.py`、`main03.py` 的数据生成、HMM/机器学习/配置求解和输出逻辑。

## 1. 主文件

- `bank_allocation_system.py`：完整系统代码，单文件可运行，也可作为 Python 模块导入。
- `requirements_bank_allocation.txt`：核心依赖。

Python 建议版本：3.10 或更高。

## 2. 安装依赖

```bash
pip install -r requirements_bank_allocation.txt
```

核心版本不依赖 `cvxpy`、`gymnasium` 或 `stable-baselines3`。配置求解使用 `scipy.optimize.SLSQP`；强化学习部分采用治理安全的离散 Q-learning，只调节风险/资本/流动性惩罚权重，不直接生成资产权重。

## 3. 最简运行

```bash
python bank_allocation_system.py
```

默认行为：

1. 生成与原 `main.py` 口径一致的季度模拟数据；
2. 顺序执行 A1—A10；
3. 输出 CSV / JSON / PNG 到 `./system_outputs/`；
4. 默认使用较轻的可复现估计配置，Bootstrap 默认 199 次。

指定目录：

```bash
python bank_allocation_system.py --sim-dir ./sim_data --out-dir ./system_outputs
```

## 4. 研究级运行

提高门槛回归 Bootstrap 次数，并启用 A3 参数 MLE：

```bash
python bank_allocation_system.py --full --bootstrap 999
```

如正式论文最终采用 1000/2000 次 Bootstrap，可直接改为：

```bash
python bank_allocation_system.py --full --bootstrap 2000
```

## 5. 可选滚动 MPC / RL

启用多期 MPC：

```bash
python bank_allocation_system.py --mpc
```

启用治理安全 RL 惩罚权重校准：

```bash
python bank_allocation_system.py --rl
```

两者同时启用：

```bash
python bank_allocation_system.py --mpc --rl
```

说明：`--mpc` 对每一期执行多期非线性约束优化，运行明显慢于默认单期增量求解；正式回测建议在高性能环境中运行，或在 `AllocationConfig` 中适当调整 `mpc_horizon`、`solver_maxiter`。

## 6. 使用真实数据

准备好以下 4 个文件后，可以跳过模拟数据生成：

```bash
python bank_allocation_system.py --reuse-data --sim-dir ./my_real_data --out-dir ./real_outputs
```

### `panel_quarterly.csv`

至少需要以下字段：

- `period`, `date`
- `A_total_assets_rmb_bn`, `g_asset_growth_q`
- `LPR1_pct`, `LPR5_pct`, `DR007_pct`
- `rA_avg_earning_asset_yield_pct`, `rL_avg_interest_bearing_cost_pct`, `NIM_state_pct`
- `alpha_mkt_funding_share_proxy`, `r_dep_deposit_cost_proxy_pct`
- `CAR_pct`, `LCR_pct`, `NSFR_pct`
- `non_interest_income_proxy_rmb_bn`, `provision_proxy_rmb_bn`, `operating_cost_proxy_rmb_bn`
- `w_credit`, `w_bond`, `w_ib`, `w_trading`, `w_cb_gov`, `w_wm_fund`

### `yc_gov_long.csv`

字段：`period`, `date`, `maturity_year`, `yld_pct`。

### `yc_cred_long.csv`

字段：`period`, `date`, `maturity_year`, `yld_pct`。

### `bucket_params.csv`

字段：

- `bucket`
- `omega_rwa_density`
- `hqla_ratio`
- `haircut`
- `rsf`
- `duration_year`

六类资产名称必须为：`credit`, `bond`, `ib`, `trading`, `cb_gov`, `wm_fund`。

## 7. A1—A10 对应关系

- A1：Hansen 单门槛回归、Bootstrap sup-F、LR 反演置信区间、顺序双门槛检验。
- A2：Granger 因果检验，全样本/低利率阶段对比，AIC/BIC 滞后选择。
- A3：多期限收益率曲线状态空间模型、Kalman 滤波/RTS 平滑、影子短端利率。
- A4：分阶段 Local Projection / 分布滞后响应，支持 HAC 标准误。
- A5：资产负债重定价桶、利息收入/支出/NII/NIM/ROE 弹性情景。
- A6：HMM 状态识别、Bayesian/Kalman 漂移、分位数预测、监管约束校准、稳健配置、存量—增量双轨、配置带宽、局部影子价格、MPC、RL 惩罚校准。
- A7：LPR1—DR007—10Y 国债外部利率阶段识别与 NIM 内部校准、情景切换。
- A8：RORAC、RORWA、资本占用强度、相对配置方向指数。
- A9：存量资产分池、新增资产五道准入门、低效资产识别与迁移。
- A10：多维回测、三类触发信号、纠偏动作、滚动参数更新。

## 8. 主要输出

输出目录中会生成：

- `A1_threshold_summary.json`, `A1_threshold_grid.csv`
- `A2_granger_full_vs_stages.csv`
- `A3_shadow_rate.csv`, `A3_fit_diagnostics.csv`, `A3_compare_diagnostics.json`
- `A4_stage_local_projection.csv`
- `A5_scenario_table.csv`, `A5_bucket_details.json`
- `A6_allocation_path.csv`, `A6_bandwidth_latest.csv`, `A6_shadow_prices_latest.json`, `A6_hmm_transition.csv`
- `A7_rate_stage.csv`
- `A8_rorac_rorwa.csv`
- `A9_stock_pool.csv`, `A9_new_asset_gates.csv`
- `A10_backtest.csv`, `A10_trigger_and_correction.csv`
- 若干诊断 PNG 图。

## 9. 关键实现口径

1. **监管指标约束**：由当期 CAR/LCR/NSFR 及实际组合反推资本额、净现金流出和可用稳定资金，再在候选组合上重新计算指标，避免使用固定比例的粗代理。
2. **存量—增量双轨**：新增资产结构 `x_t` 只通过当期新增资产占比逐步改变总资产权重，不允许模型假设存量组合可以瞬时重构。
3. **RL 治理边界**：RL 只选择 `lambda_var/lambda_cap/lambda_liq` 的惩罚倍数；最终资产权重始终由带硬约束的优化器求解。
4. **小样本保护**：Granger、LP、分位数预测等在有效样本不足时返回明确的 `insufficient sample`/空表，不强行输出统计结论。
5. **数据性质**：A8/A9 对估算值与公开值可保留 `DataFlag`，真实数据替换时建议沿用这一字段。

## 10. 作为模块调用

```python
from bank_allocation_system import SystemConfig, run_system

cfg = SystemConfig()
result = run_system(cfg, force_generate=True)
```

也可以单独导入 A1—A10 的函数，用于论文复现或单模块检验。

numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
statsmodels>=0.14
matplotlib>=3.7

