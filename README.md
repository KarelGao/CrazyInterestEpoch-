# CrazyInterestEpoch-

本项目构建一套面向商业银行低利率环境的大类资产配置决策系统。系统整合门槛回归、Granger 因果检验、状态空间模型、局部投影、盈利弹性测算、隐马尔可夫模型（HMM）、机器学习预测、模型预测控制（MPC）与强化学习（RL），形成“利率状态识别—盈利压力测算—六类资产配置优化—资本与流动性约束校验—存量/增量资产准入—回测与动态纠偏”的完整闭环。

模型以 CAR、LCR、NSFR、RWA、久期和交易资产上限等监管与风险指标约束可行域，输出资产配置区间、增量调整方向、约束边界、KKT 影子价格与滚动优化结果，可用于银行资产负债管理和大类资产配置研究。

# NIM-State Robust Allocation Research System（A1—A10）

本程序中的 A1—A10 算法严格按编号拆分为十个 Python 模块，并复用、重构数据生成、HMM、机器学习、MPC、PPO 和输出逻辑。

## 1. 工程结构

正式版本采用 A1—A10 模块化结构：

- `A1_threshold_regression.py`：Hansen 门槛回归与 Bootstrap 显著性检验。
- `A2_granger_causality.py`：全样本与低利率阶段 Granger 因果检验。
- `A3_shadow_rate.py`：多期限收益率曲线状态空间模型与影子利率。
- `A4_local_projection.py`：分阶段 Local Projection 动态响应。
- `A5_profitability_elasticity.py`：资产负债重定价与盈利弹性测算。
- `A6_allocation_engine.py`：NIM-State Robust 核心配置引擎。
- `A7_rate_stage_recognition.py`：外部利率阶段识别与 NIM 内部校准。
- `A8_RORAC_RORWA.py`：RORAC / RORWA 风险调整资本效率诊断。
- `A9_stock_new_asset_gates.py`：存量资产分层与新增资源准入。
- `A10_backtest_rolling_calibration.py`：回测、触发与滚动纠偏。
- `main.py`：总控程序，严格按 A1 → A10 顺序执行。
- `config.py`：统一参数与模块开关。
- `common.py`：公共数学与数据处理函数。
- `data_generator.py`：季度模拟数据生成器。
- `outputs.py`：中文结果表和诊断图输出。
- `requirements.txt`：完整依赖。
- `check_environment.py`：检查 CVXPY、OSQP/ECOS、Gymnasium、SB3、PyTorch 等运行环境。

Python 建议版本：3.10 或更高。

## 2. 安装依赖

```bash
pip install -r requirements.txt
python check_environment.py
```

A6 正式核心配置模块采用 **CVXPY** 构建硬约束单期配置与多期 MPC，并优先使用 **OSQP / ECOS** 完成凸优化求解。强化学习模块基于 **Gymnasium + Stable-Baselines3 PPO**，底层使用 PyTorch。

PPO 只动态校准：

- `lambda_var`：风险惩罚权重；
- `lambda_cap`：资本占用惩罚权重；
- `lambda_liq`：流动性与稳定资金惩罚权重。

PPO **不直接生成任何资产权重**。六类资产权重始终由带 CAR/LCR/NSFR、久期、交易限额和资产区间硬约束的 CVXPY 优化问题求解。

## 3. 最简运行

完整模型默认启用 MPC 和 PPO：

```bash
python main.py
```

默认行为：

1. 生成与原 `main.py` 口径一致的季度模拟数据；
2. 严格按照 A1 → A10 执行；
3. A6 默认启用 CVXPY 多期 MPC；
4. A6 默认启用 PPO 惩罚参数校准；
5. Bootstrap 默认 199 次；
6. PPO 默认训练 1000 timesteps；
7. 输出 CSV / JSON / PNG 到 `./system_outputs/`。

指定目录：

```bash
python main.py --sim-dir ./sim_data --out-dir ./system_outputs
```

## 4. 研究级运行

提高 Bootstrap 次数、启用 A3 参数 MLE，并提高 PPO 训练步数：

```bash
python main.py --full --bootstrap 1000 --ppo-timesteps 10000
```

如果最终论文采用 2000 次 Bootstrap：

```bash
python main.py --full --bootstrap 2000 --ppo-timesteps 20000
```

## 5. MPC / PPO 消融与稳健性检验

完整模型默认启用 MPC 和 PPO。以下开关仅用于消融实验、稳健性检验或调试。

关闭 PPO、保留 CVXPY MPC：

```bash
python main.py --no-rl
```

关闭多期 MPC、保留 PPO，并使用单期 CVXPY 增量配置：

```bash
python main.py --no-mpc
```

同时关闭多期 MPC 与 PPO：

```bash
python main.py --no-mpc --no-rl
```

即使关闭多期 MPC，A6 仍然使用 CVXPY 进行单期硬约束配置求解；不会退回 SLSQP 作为正式配置算法。

## 6. 使用真实数据

准备以下 4 个文件后，可以跳过模拟数据生成：

```bash
python main.py --reuse-data --sim-dir ./my_real_data --out-dir ./real_outputs
```

### `panel_quarterly.csv`

至少需要：

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

六类资产名称固定为：

`credit`, `bond`, `ib`, `trading`, `cb_gov`, `wm_fund`。

## 7. A1—A10 对应关系

- **A1**：Hansen 单门槛回归、Bootstrap sup-F、LR 反演置信区间、顺序双门槛检验。
- **A2**：Granger 因果检验，全样本/低利率阶段对比，AIC/BIC 滞后选择。
- **A3**：多期限收益率曲线状态空间模型、Kalman 滤波/RTS 平滑、影子短端利率。
- **A4**：分阶段 Local Projection / 分布滞后响应，支持 HAC 标准误。
- **A5**：资产负债重定价桶、利息收入/支出、NII/NIM/ROE 弹性情景。
- **A6**：HMM 状态识别、Bayesian/Kalman 参数漂移、分位数预测、CVXPY 硬约束配置、存量—增量双轨、多期 MPC、配置带宽、KKT 影子价格与 PPO 惩罚权重校准。
- **A7**：LPR1—DR007—10Y 国债外部利率阶段识别与 NIM 内部校准、情景切换。
- **A8**：RORAC、RORWA、资本占用强度和相对配置方向指数。
- **A9**：存量资产分池、新增资产五道准入门、低效资产识别与迁移。
- **A10**：多维回测、三类触发信号、纠偏动作和滚动参数更新。

## 8. A6 正式技术路线

A6 的完整执行链为：

```text
HMM 状态识别
    ↓
Bayesian / Kalman 参数漂移
    ↓
Quantile ML 约束预测
    ↓
PPO 参数校准
    ├─ lambda_var
    ├─ lambda_cap
    └─ lambda_liq
    ↓
CVXPY 多期 MPC
    ├─ CAR 硬约束
    ├─ LCR 硬约束
    ├─ NSFR 硬约束
    ├─ 久期硬约束
    ├─ 交易资产上限
    └─ 六类资产治理带宽
    ↓
第一期最优增量配置 x_t
    ↓
存量继承后的总资产权重 w_t
    ↓
KKT dual value / 约束影子价格
```

CAR、LCR、NSFR 在 CVXPY 中通过代数变换写为线性硬约束：

- CAR：`omega @ w <= capital / (A * CAR_floor)`；
- LCR：`hqla @ w >= (NCO / A) * LCR_floor`；
- NSFR：`rsf @ w <= (ASF / A) / NSFR_floor`。

因此可以直接从对应 CVXPY constraint 的 `dual_value` 读取约束影子价格，而不再通过局部数值扰动近似。

## 9. 主要输出

输出目录中包括：

- `A1_threshold_summary.json`, `A1_threshold_grid.csv`
- `A2_granger_full_vs_stages.csv`
- `A3_shadow_rate.csv`, `A3_fit_diagnostics.csv`, `A3_compare_diagnostics.json`
- `A4_stage_local_projection.csv`
- `A5_profitability_elasticity.csv`, `A5_plot_data.csv`, `A5_bucket_details.json`
- `A6_allocation_path.csv`
- `A6_bandwidth_latest.csv`
- `A6_shadow_prices_latest.json`
- `A6_hmm_transition.csv`
- `A6_model_diagnostics.json`
- `A6_ppo_penalty_tuner.zip`（PPO 开启时）
- `A7_rate_stage_and_nim_calibration.csv`
- `A8_RORAC_RORWA_diagnostics.csv`
- `A9_stock_pool_and_gates.csv`
- `A10_backtest.csv`, `A10_trigger_and_correction.csv`
- `table_overview_cn.csv`, `table_direction_cn.csv`
- 若干诊断 PNG 图。

## 10. 关键实现口径

1. **监管约束重构**：由当期 CAR/LCR/NSFR 和实际组合反推资本额、净现金流出及可用稳定资金，再对候选组合重新计算监管指标。
2. **存量—增量双轨**：新增资产结构 `x_t` 仅通过当期新增资产占比逐步改变总资产权重，存量组合不能瞬时重构。
3. **MPC**：在滚动预测区间内联合优化未来多期新增资产结构，只执行第一期控制量，下一期重新估计并滚动求解。
4. **PPO 治理边界**：PPO 只有三个惩罚参数动作，不存在六类资产权重动作；资产权重始终由 CVXPY 硬约束优化器决定。
5. **影子价格**：CAR/LCR/NSFR 等约束影子价格直接来自 CVXPY KKT 对偶变量 `dual_value`。
6. **小样本保护**：Granger、LP、分位数预测等在有效样本不足时返回明确的空结果/样本不足状态，不强行生成统计结论。
7. **数据性质**：A8/A9 可保留 `DataFlag` 区分公开值与估算值，真实数据替换时建议继续沿用。

## 11. 作为 Python 模块调用

完整系统：

```python
from config import SystemConfig
from main import run_system

cfg = SystemConfig()
result = run_system(cfg, force_generate=True)
```

单独调用 A6：

```python
from config import AllocationConfig
from A6_allocation_engine import run_a6

cfg = AllocationConfig()
result_a6 = run_a6(panel, yc_gov, yc_cred, bucket_params, cfg)
```

单独调用 A1—A10 时，各文件均保留 `run_a1`、`run_a2` …… `run_a10` 的编号化公共入口。

## 12. 方法一致性说明

正式版不再以 SLSQP 或离散表格 Q-learning 作为 A6 核心方法。完整研究模型的 A6 主路径固定为：

**HMM + Bayesian/Kalman Drift + Quantile ML + PPO + CVXPY MPC + OSQP/ECOS + KKT Duals**。

单期 CVXPY、关闭 PPO 等模式仅用于消融、稳健性检验和故障定位，不改变正式研究方法口径。
