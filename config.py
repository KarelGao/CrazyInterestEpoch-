# -*- coding: utf-8 -*-
"""Central configuration objects. Algorithm files remain A1-A10 only."""
from __future__ import annotations
from dataclasses import dataclass, field

class ThresholdConfig:
    trimming: float = 0.10
    bootstrap_B: int = 199
    alpha_list: tuple[float, ...] = (0.10, 0.05, 0.01)
    fe_flag: str = "none"           # none/time/entity/twoway
    boot_type: str = "wild"         # residual/wild
    grid_type: str = "quantileGrid" # uniqueQ/quantileGrid
    grid_points: int = 81
    seed: int = 2026
    double_threshold: bool = True


@dataclass
class AllocationConfig:
    # Hard regulatory / governance floors
    car_min: float = 12.0
    lcr_min: float = 120.0
    nsfr_min: float = 105.0
    duration_cap: float = 4.5
    trading_cap: float = 0.15

    # Six-bucket governance bands
    w_lower: tuple[float, ...] = (0.15, 0.10, 0.02, 0.00, 0.02, 0.02)
    w_upper: tuple[float, ...] = (0.70, 0.55, 0.25, 0.15, 0.20, 0.25)

    # Base penalty coefficients. PPO may tune only the first three.
    lambda_var: float = 0.35
    lambda_cap: float = 0.50
    lambda_liq: float = 0.50
    lambda_turn: float = 0.10
    kappa_lcr: float = 1.0
    kappa_nsfr: float = 1.0
    capital_shadow_base: float = 0.02
    liquidity_shadow_base: float = 0.02

    # Safety buffers above hard regulatory floors
    safety_car_margin: float = 0.25
    safety_lcr_margin: float = 5.0
    safety_nsfr_margin: float = 1.5

    # MPC settings
    discount: float = 0.97
    mpc_horizon: int = 4
    use_mpc: bool = True

    # Model switches (for ablation / robustness tests)
    use_hmm: bool = True
    use_bayes: bool = True
    use_quantile: bool = True
    use_rl: bool = True

    # Quantile and bandwidth settings
    q_delta: float = 0.10
    bandwidth_rho: float = 1e-4

    # CVXPY / solver settings
    cvxpy_solvers: tuple[str, ...] = ("OSQP", "ECOS")
    solver_maxiter: int = 20000
    solver_eps_abs: float = 1e-7
    solver_eps_rel: float = 1e-7
    solver_verbose: bool = False

    # PPO settings. PPO actions tune lambda_var/lambda_cap/lambda_liq only.
    ppo_timesteps: int = 1000
    ppo_n_steps: int = 64
    ppo_batch_size: int = 64
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.95
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.20
    ppo_ent_coef: float = 0.0
    ppo_lambda_min_mult: float = 0.50
    ppo_lambda_max_mult: float = 2.00
    ppo_verbose: int = 0
    ppo_device: str = "auto"

    seed: int = 2026


@dataclass
class StageConfig:
    persist_k: int = 2
    # Stage score is ordinal. Lower rate -> larger stage index.
    lpr_centres: tuple[float, ...] = (4.0, 3.0, 2.0, 1.0, 0.0, -0.5)
    dr_centres: tuple[float, ...] = (3.0, 2.3, 1.7, 1.2, 0.5, -0.2)
    y10_centres: tuple[float, ...] = (3.5, 2.8, 2.2, 1.6, 0.8, -0.2)
    weights: tuple[float, float, float] = (0.60, 0.20, 0.20)
    nim_thresholds: tuple[float, float, float] = (1.90, 1.80, 1.70)
    divergence_tolerance: float = 0.50


@dataclass
class SystemConfig:
    seed: int = 20260213
    start: str = "2015Q1"
    end: str = "2025Q4"
    out_dir: str = "./system_outputs"
    sim_dir: str = "./sim_data"
    fast_mode: bool = True
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    # Trigger thresholds are execution inputs, not embedded in A10 itself.
    trigger_set: dict[str, dict[str, dict[str, float | str]]] = field(default_factory=lambda: {
        "External": {
            "LPR1_change": {"type": "abs_change_ge", "value": 0.20},
            "DR007_change": {"type": "abs_change_ge", "value": 0.15},
            "Y10_change": {"type": "abs_change_ge", "value": 0.20},
        },
        "Operating": {
            "NIM": {"type": "min", "value": 1.80},
            "NIM_change": {"type": "change_le", "value": -0.05},
        },
        "Constraint": {
            "CAR": {"type": "min", "value": 12.0},
            "LCR": {"type": "min", "value": 120.0},
            "NSFR": {"type": "min", "value": 105.0},
        },
    })
