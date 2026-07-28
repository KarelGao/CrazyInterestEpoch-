# -*- coding: utf-8 -*-
"""Data generator / input adapter for the A1-A10 research system.

This file is support code, not an appendix algorithm module.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator, x0: float = 0.0) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    x[0] = x0
    eps = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def simulate_nim_state_data(
    start: str = "2015Q1",
    end: str = "2025Q4",
    seed: int = 42,
    out_dir: str = "./sim_data",
    leave_blank_other_assets: bool = True,
    blank_share_range: tuple[float, float] = (0.03, 0.08),  # leave 3%~8% as "other assets"
    lpr_scenario: str = "trend",  # 'trend' | 'stage'
    lpr_stage_levels: tuple[float, ...] = (3.0, 2.5, 2.0, 1.5),
    lpr_stage_len: int = 8,  # quarters per stage (only for lpr_scenario='stage')
) -> dict[str, pd.DataFrame]:
    """
    Generate simulated data meeting the pseudocode input requirements.

    Parameters
    ----------
    start/end : quarterly period labels, e.g. '2015Q1'
    seed      : random seed for reproducibility
    out_dir   : folder to save outputs
    leave_blank_other_assets : if True, six bucket weights sum to <=1
    blank_share_range : range of "other assets" share (only used if leave_blank_other_assets=True)
    lpr_scenario : 'trend' (historical-style) or 'stage' (3->2.5->2->1.5)
    lpr_stage_levels : stage levels for LPR1 (only for lpr_scenario='stage')
    lpr_stage_len : quarters per stage

    Returns
    -------
    dict of DataFrames: panel, yc_gov_long, yc_cred_long, bucket_params, weights
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 0) Time index
    # ------------------------------------------------------------
    periods = pd.period_range(start=start, end=end, freq="Q")
    n = len(periods)
    dates = periods.to_timestamp(how="end")

    # ------------------------------------------------------------
    # 1) Macro rates: LPR1, LPR5, DR007
    # ------------------------------------------------------------
    t = np.arange(n)

    # Trend helper: monotone-ish decline with mild undulation
    def _decline_series(start_level, end_level, noise=0.03, cyc=0.05):
        base = np.linspace(start_level, end_level, n)
        wave = cyc * np.sin(2*np.pi*t/16.0)  # ~4-year cycle
        ar = _ar1(n, phi=0.6, sigma=noise, rng=rng, x0=0.0)
        return base + wave + ar

    # Stage helper: piecewise-constant levels with mild jitter + smoothing
    def _build_stage_series(stage_levels, stage_len, n, rng, jitter=0.015, smooth=3):
        x = []
        for lv in stage_levels:
            x += [lv] * int(stage_len)
        x = np.array(x[:n], float)
        if x.size < n:
            x = np.pad(x, (0, n - x.size), mode="edge")
        x = x + rng.normal(0.0, float(jitter), size=n)
        if smooth and int(smooth) > 1:
            x = pd.Series(x).rolling(int(smooth), min_periods=1).mean().to_numpy()
        return x

    if lpr_scenario.lower() == "stage":
        # Explicit scenario: 3.0 -> 2.5 -> 2.0 -> 1.5 (each lpr_stage_len quarters by default)
        LPR1 = _build_stage_series(lpr_stage_levels, lpr_stage_len, n, rng, jitter=0.012, smooth=3)

        # LPR5 keeps a term premium over LPR1 (slightly time-varying)
        term_premium = 0.55 + 0.08*np.sin(2*np.pi*t/18.0) + _ar1(n, 0.5, 0.03, rng)
        LPR5 = np.clip(LPR1 + term_premium, 1.80, 4.50)

        # DR007 moves below LPR1; clamp tail into a tighter band for realism
        dr_spread = 1.10 + 0.12*np.sin(2*np.pi*t/14.0) + _ar1(n, 0.4, 0.05, rng)
        DR007 = np.clip(LPR1 - dr_spread + rng.normal(0, 0.04, n), 0.80, 3.20)
        if n >= 6:
            tail = rng.normal(0.0, 0.04, size=6)
            DR007[-6:] = np.clip(1.55 + tail, 1.35, 1.75)
    else:
        # Default: China-like downward trend + cycles (historical-style)
        # LPR levels (approx reality: higher in 2015-2016, lower by 2024-2025)
        LPR1 = _decline_series(4.90, 3.10, noise=0.04, cyc=0.06)
        LPR5 = _decline_series(5.50, 3.60, noise=0.05, cyc=0.07)

        # DR007: higher/volatile earlier, lower and stable around 1.4~1.8 in recent years
        DR_base = _decline_series(2.80, 1.55, noise=0.06, cyc=0.10)
        # add occasional liquidity shocks
        shock = np.zeros(n)
        shock_idx = rng.choice(np.arange(2, n-2), size=max(2, n//18), replace=False)
        shock[shock_idx] += rng.normal(0.20, 0.08, size=len(shock_idx))
        DR007 = np.clip(DR_base + shock, 0.90, 4.50)

        # Enforce end-period realism (recent DR007 tight band)
        # last 6 quarters clamp gently toward 1.4~1.7 with small noise
        if n >= 6:
            tail = rng.normal(0.0, 0.04, size=6)
            DR007[-6:] = np.clip(1.55 + tail, 1.35, 1.75)

    # ------------------------------------------------------------
    # 2) Yield curves: gov + high-grade credit curve by maturity
    #    Store as long format: (date, maturity_year, yld)
    # ------------------------------------------------------------
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10], dtype=float)

    # Gov curve: level tied to (LPR1/5 mix) + term premium; slope varies with cycle
    level = 0.55 * LPR1 + 0.45 * LPR5 - 1.10  # bring gov yields below LPR
    slope = 0.55 + 0.20 * np.sin(2*np.pi*t/14.0) + _ar1(n, 0.4, 0.05, rng)  # term premium
    curvature = 0.12 * np.sin(2*np.pi*t/10.0)

    yc_gov = np.zeros((n, len(maturities)))
    for j, m in enumerate(maturities):
        # Nelson-Siegel-ish shape (simple)
        tau = 2.0
        load1 = (1 - np.exp(-m/tau)) / (m/tau)
        load2 = load1 - np.exp(-m/tau)
        y = level + slope * load1 + curvature * load2 + rng.normal(0.0, 0.03, size=n)
        yc_gov[:, j] = np.clip(y, 0.80, 5.50)

    # Credit curve: gov + spread (spread widens when NIM pressure/regime hardening)
    spread_base = 0.85 + 0.15*np.sin(2*np.pi*t/12.0) + _ar1(n, 0.5, 0.06, rng)

    # Stage/low-rate pressure proxy: higher when policy rate is lower
    press = _sigmoid(2.8 * (2.4 - LPR1))
    spread_base = spread_base + 0.12 * press  # mild widening under stronger NIM pressure

    # allow mild tightening recently
    if n >= 8:
        spread_base[-8:] -= np.linspace(0.05, 0.12, 8)

    yc_cred = np.zeros_like(yc_gov)
    for j, m in enumerate(maturities):
        term_spread = spread_base + 0.10*np.log1p(m)  # longer tenor a bit wider
        yc_cred[:, j] = np.clip(yc_gov[:, j] + term_spread, 1.20, 8.50)

    yc_gov_long = pd.DataFrame({
        "period": np.repeat(periods.astype(str), len(maturities)),
        "date": np.repeat(dates, len(maturities)),
        "maturity_year": np.tile(maturities, n),
        "yld_pct": yc_gov.reshape(-1)
    })

    yc_cred_long = pd.DataFrame({
        "period": np.repeat(periods.astype(str), len(maturities)),
        "date": np.repeat(dates, len(maturities)),
        "maturity_year": np.tile(maturities, n),
        "yld_pct": yc_cred.reshape(-1)
    })

    # ------------------------------------------------------------
    # 3) Six buckets meta: omega (risk weight), h (haircut), rsf, D (duration proxy)
    # ------------------------------------------------------------
    buckets = ["credit", "bond", "ib", "trading", "cb_gov", "wm_fund"]
    omega = np.array([0.75, 0.25, 0.20, 0.35, 0.00, 0.50], float)   # RWA density proxy
    hqla_ratio = np.array([0.05, 0.45, 0.20, 0.05, 0.90, 0.05], float)  # HQLA eligibility proxy
    haircut = np.array([0.08, 0.04, 0.02, 0.12, 0.01, 0.10], float)  # market haircut proxy
    rsf = np.array([0.85, 0.20, 0.10, 0.30, 0.05, 0.70], float)      # NSFR RSF proxy
    D = np.array([2.2, 4.8, 0.3, 1.2, 2.0, 2.8], float)              # duration proxy (years)

    bucket_params = pd.DataFrame({
        "bucket": buckets,
        "omega_rwa_density": omega,
        "hqla_ratio": hqla_ratio,
        "haircut": haircut,
        "rsf": rsf,
        "duration_year": D,
    })

    # ------------------------------------------------------------
    # 4) Balance sheet scale A_t and growth g_t (RMB bn)
    # ------------------------------------------------------------
    A0 = 1200.0  # starting total assets in RMB bn (placeholder scale)
    g_q = 0.022 + 0.005*np.sin(2*np.pi*t/12.0) + _ar1(n, 0.3, 0.004, rng)
    g_q = np.clip(g_q, -0.01, 0.06)
    A = A0 * np.cumprod(1.0 + g_q)

    # ------------------------------------------------------------
    # 5) Six-bucket weights W_t (sum to <=1 if leave_blank_other_assets=True)
    # ------------------------------------------------------------
    # "other assets" blank share
    if leave_blank_other_assets:
        blank = rng.uniform(blank_share_range[0], blank_share_range[1], size=n)
    else:
        blank = np.zeros(n)

    # baseline bucket shares (rough China-like) + mild dynamics
    w_credit = 0.46 - 0.06*_sigmoid((4.0 - LPR1)) + 0.02*np.sin(2*np.pi*t/18.0)
    w_bond   = 0.22 + 0.05*_sigmoid((4.0 - LPR1)) + 0.01*np.sin(2*np.pi*t/20.0)
    w_ib     = 0.10 + 0.02*np.sin(2*np.pi*t/12.0)
    w_trd    = 0.05 + 0.01*np.sin(2*np.pi*t/10.0)
    w_cb     = 0.06 + 0.01*np.sin(2*np.pi*t/16.0)
    w_wm     = 0.07 + 0.02*_sigmoid((3.6 - LPR1)) + 0.01*np.sin(2*np.pi*t/14.0)

    W_raw = np.vstack([w_credit, w_bond, w_ib, w_trd, w_cb, w_wm]).T
    W_raw = np.clip(W_raw, 0.01, None)

    # normalize to (1-blank) each period
    W = (W_raw / W_raw.sum(axis=1, keepdims=True)) * (1.0 - blank.reshape(-1, 1))
    W = np.clip(W, 0.0, 1.0)

    weights = pd.DataFrame(W, columns=[f"w_{b}" for b in buckets])
    weights.insert(0, "period", periods.astype(str))
    weights.insert(1, "date", dates)

    # ------------------------------------------------------------
    # 6) Funding mix / repricing proxies: alpha_t, r_dep_t
    #    alpha_t: market-based funding share proxy (rises mildly over time, cyclical)
    #    deposit proxy: moves slower than DR007; anchored to policy rates
    # ------------------------------------------------------------
    alpha = 0.30 + 0.06*np.sin(2*np.pi*t/18.0) + np.linspace(0.00, 0.05, n) + _ar1(n, 0.3, 0.01, rng)
    alpha = np.clip(alpha, 0.18, 0.48)

    # deposit repricing proxy (pct): sticky; linked to LPR1 with longer lag + dynamic floor
    dep_base = 0.62 * LPR1 - 0.70  # slightly higher baseline vs prior
    dep_slow = pd.Series(dep_base).rolling(6, min_periods=1).mean().to_numpy()  # longer lag (was 4)
    dep_floor = 0.95 + 0.10 * _sigmoid(3.0 * (2.2 - LPR1))  # floor rises mildly when LPR gets very low
    dep_noise = rng.normal(0.0, 0.025, size=n)
    r_dep = np.clip(dep_slow + dep_noise, dep_floor, 3.60)

    # liability cost from mechanism (this becomes rL_t)
    rL = alpha * DR007 + (1.0 - alpha) * r_dep
    rL = np.clip(rL, 0.80, 4.20)

    # ------------------------------------------------------------
    # 7) Bucket yields r_{i,t} and avg earning-asset yield rA_t
    #    r_{i,t} driven by curves + spreads; credit linked to LPR; bonds to gov/cred curve
    # ------------------------------------------------------------
    # pick representative tenors
    gov_5y = yc_gov[:, np.where(maturities == 5)[0][0]]
    gov_10y = yc_gov[:, np.where(maturities == 10)[0][0]]
    cred_3y = yc_cred[:, np.where(maturities == 3)[0][0]]
    cred_5y = yc_cred[:, np.where(maturities == 5)[0][0]]

    # Loss/haircut proxy: higher when rates volatile or credit spread widens
    vol_proxy = np.abs(np.diff(gov_10y, prepend=gov_10y[0]))
    spread_proxy = (cred_5y - gov_5y)
    loss_common = 0.06 * _sigmoid(8*(vol_proxy - np.median(vol_proxy))) + 0.05 * _sigmoid(3*(spread_proxy - np.median(spread_proxy)))

    # bucket yields (pct)
    r_credit = np.clip(0.92*LPR1 + 0.10*LPR5 - 0.25 + rng.normal(0, 0.08, n) - 0.20*loss_common, 2.00, 6.50)
    r_bond   = np.clip(0.55*cred_5y + 0.45*gov_10y + 0.15 + rng.normal(0, 0.06, n) - 0.10*loss_common, 1.50, 6.50)
    r_ib     = np.clip(DR007 + 0.20 + rng.normal(0, 0.05, n), 1.00, 5.50)
    r_trd    = np.clip(0.55*cred_3y + 0.25*DR007 + 0.20*gov_5y + 0.10 + rng.normal(0, 0.12, n) - 0.35*loss_common, 1.00, 8.00)
    r_cb     = np.clip(0.80*gov_5y + 0.10 + rng.normal(0, 0.04, n), 1.00, 4.80)
    r_wm     = np.clip(0.55*cred_5y + 0.25*LPR1 + 0.20*gov_5y + 0.35 + rng.normal(0, 0.10, n) - 0.25*loss_common, 1.50, 7.50)

    # average earning-asset yield rA_t from weights
    w_mat = W.copy()
    rA = (
        w_mat[:, 0]*r_credit
        + w_mat[:, 1]*r_bond
        + w_mat[:, 2]*r_ib
        + w_mat[:, 3]*r_trd
        + w_mat[:, 4]*r_cb
        + w_mat[:, 5]*r_wm
    )
    # scale to plausible bank-level range
    rA = np.clip(rA / np.clip(w_mat.sum(axis=1), 0.85, 1.0), 2.30, 5.20)

    # NIM state
    NIM = np.clip(rA - rL, 0.85, 3.10)

    # ------------------------------------------------------------
    # 8) Regulatory metrics (CAR/LCR/NSFR) + implied NCO/ASF for mapping
    # ------------------------------------------------------------
    # CAR around 12~14.5 with mild down pressure when NIM is low
    CAR = 13.6 + 0.35*np.sin(2*np.pi*t/20.0) + _ar1(n, 0.4, 0.15, rng) - 0.45*_sigmoid((np.median(NIM)-NIM)*2.0)
    CAR = np.clip(CAR, 11.5, 15.5)

    # HQLA proxy from weights
    HQLA = A * (w_mat[:, 0]*hqla_ratio[0] + w_mat[:, 1]*hqla_ratio[1] + w_mat[:, 2]*hqla_ratio[2] +
                w_mat[:, 3]*hqla_ratio[3] + w_mat[:, 4]*hqla_ratio[4] + w_mat[:, 5]*hqla_ratio[5])

    # LCR: relatively high (150~250), with cycles; then back out NCO = HQLA/LCR
    LCR = 190 + 35*np.sin(2*np.pi*t/14.0) + _ar1(n, 0.35, 18.0, rng)
    LCR = np.clip(LCR, 120, 280)
    NCO = HQLA / (LCR/100.0)

    # NSFR: stable 108~132; back out ASF = NSFR * RSF
    RSF_amt = A * (w_mat[:, 0]*rsf[0] + w_mat[:, 1]*rsf[1] + w_mat[:, 2]*rsf[2] +
                   w_mat[:, 3]*rsf[3] + w_mat[:, 4]*rsf[4] + w_mat[:, 5]*rsf[5])

    NSFR = 120 + 6*np.sin(2*np.pi*t/18.0) + _ar1(n, 0.35, 3.0, rng)
    NSFR = np.clip(NSFR, 105, 135)
    ASF = (NSFR/100.0) * RSF_amt

    # Optional non-interest items (kept simple but stable)
    nonint = 0.18 * A / 100.0 + rng.normal(0, 0.35, n)
    prov = 0.10 * A / 100.0 + rng.normal(0, 0.25, n)
    cost = 0.14 * A / 100.0 + rng.normal(0, 0.30, n)
    div = np.clip(0.02 * A / 100.0 + rng.normal(0, 0.05, n), 0.0, None)

    # ------------------------------------------------------------
    # 9) Assemble panel
    # ------------------------------------------------------------
    panel = pd.DataFrame({
        "period": periods.astype(str),
        "date": dates,

        "A_total_assets_rmb_bn": A,
        "g_asset_growth_q": g_q,

        "LPR1_pct": LPR1,
        "LPR5_pct": LPR5,
        "DR007_pct": DR007,

        "rA_avg_earning_asset_yield_pct": rA,
        "rL_avg_interest_bearing_cost_pct": rL,
        "NIM_state_pct": NIM,

        "alpha_mkt_funding_share_proxy": alpha,
        "r_dep_deposit_cost_proxy_pct": r_dep,

        "CAR_pct": CAR,
        "LCR_pct": LCR,
        "NSFR_pct": NSFR,

        "HQLA_proxy_rmb_bn": HQLA,
        "NCO_proxy_rmb_bn": NCO,
        "RSF_proxy_rmb_bn": RSF_amt,
        "ASF_proxy_rmb_bn": ASF,

        "non_interest_income_proxy_rmb_bn": nonint,
        "provision_proxy_rmb_bn": prov,
        "operating_cost_proxy_rmb_bn": cost,
        "dividend_proxy_rmb_bn": div,
    })

    # merge weights into panel for convenience
    panel = panel.merge(weights.drop(columns=["date"]), on="period", how="left")

    # ------------------------------------------------------------
    # 10) Save files
    # ------------------------------------------------------------
    panel_path = os.path.join(out_dir, "panel_quarterly.csv")
    yc_gov_path = os.path.join(out_dir, "yc_gov_long.csv")
    yc_cred_path = os.path.join(out_dir, "yc_cred_long.csv")
    params_path = os.path.join(out_dir, "bucket_params.csv")
    weights_path = os.path.join(out_dir, "weights.csv")
    meta_path = os.path.join(out_dir, "config_meta.json")

    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    yc_gov_long.to_csv(yc_gov_path, index=False, encoding="utf-8-sig")
    yc_cred_long.to_csv(yc_cred_path, index=False, encoding="utf-8-sig")
    bucket_params.to_csv(params_path, index=False, encoding="utf-8-sig")
    weights.to_csv(weights_path, index=False, encoding="utf-8-sig")

    meta = {
        "seed": seed,
        "start": start,
        "end": end,
        "freq": "Q",
        "maturities_year": maturities.tolist(),
        "leave_blank_other_assets": leave_blank_other_assets,
        "blank_share_range": list(blank_share_range),
        "lpr_scenario": lpr_scenario,
        "lpr_stage_levels": list(lpr_stage_levels),
        "lpr_stage_len": int(lpr_stage_len),
        "notes": {
            "units": {
                "rates": "percent",
                "assets": "RMB billion",
                "LCR/NSFR/CAR": "percent",
            },
            "six_bucket_sum_rule": "sum(w_*) <= 1 when leave_blank_other_assets=True"
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "panel": panel,
        "yc_gov_long": yc_gov_long,
        "yc_cred_long": yc_cred_long,
        "bucket_params": bucket_params,
        "weights": weights,
    }
