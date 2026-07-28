# -*- coding: utf-8 -*-
"""Output tables and figures; not part of A1-A10 algorithm numbering."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK=True
except Exception:
    MATPLOTLIB_OK=False
from common import BUCKETS, BUCKET_CN

def _plot_outputs(outdir:Path,panel:pd.DataFrame,stage:pd.DataFrame,a3:dict[str,Any],a4:pd.DataFrame,a6:dict[str,Any]):
    if not MATPLOTLIB_OK:return
    # LPR/NIM trend
    fig,ax=plt.subplots(figsize=(10,4.5)); x=np.arange(len(panel)); ax.plot(x,panel["LPR1_pct"],label="1Y LPR (%)"); ax.plot(x,panel["NIM_state_pct"],label="NIM (%)"); ax.set_title("1Y LPR and NIM"); ax.set_xlabel("Quarter"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(outdir/"fig_lpr_nim.png",dpi=180); plt.close(fig)
    # Shadow rate
    fig,ax=plt.subplots(figsize=(10,4.5)); ax.plot(x,a3["rShadow"],label="Shadow short rate"); ax.plot(x,panel["DR007_pct"],label="DR007"); ax.set_title("Shadow-rate proxy vs DR007"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(outdir/"fig_shadow_rate.png",dpi=180); plt.close(fig)
    # LP
    fig,ax=plt.subplots(figsize=(8,4.5))
    for st,g in a4.groupby("Stage"):
        ax.plot(g["Horizon"],g["IRF"],marker="o",label=st)
    ax.axhline(0,lw=.8); ax.set_title("Stage-specific NIM response"); ax.set_xlabel("Horizon"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(outdir/"fig_local_projection.png",dpi=180); plt.close(fig)
    # allocation latest
    hist=a6["history"]; w=hist[[f"w_mpc_{b}" for b in BUCKETS]].iloc[-1]
    fig,ax=plt.subplots(figsize=(9,4.5)); ax.bar(BUCKETS,w.to_numpy()); ax.set_title("Latest feasible allocation weights"); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(outdir/"fig_latest_allocation.png",dpi=180); plt.close(fig)
