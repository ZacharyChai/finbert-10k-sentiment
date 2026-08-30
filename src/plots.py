"""Figures for the tone / forward-return analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
FB = "#1d3557"
LM = "#c1121f"


def plot_tone_over_time(by_year: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(by_year.index, by_year["fb_pct_neg"], "o-", color=FB,
             label="FinBERT: % sentences negative")
    ax1.set_ylabel("FinBERT share negative", color=FB)
    ax1.tick_params(axis="y", labelcolor=FB)
    ax2 = ax1.twinx()
    ax2.plot(by_year.index, by_year["lm_neg"], "s--", color=LM,
             label="Loughran-McDonald: negative-word fraction")
    ax2.set_ylabel("LM negative-word fraction", color=LM)
    ax2.tick_params(axis="y", labelcolor=LM)
    ax2.grid(False)
    ax1.set_title("10-K Risk Factors tone over time (universe mean)")
    ax1.set_xlabel("fiscal year")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_finbert_vs_lm(df: pd.DataFrame, path: Path) -> None:
    d = df.dropna(subset=["fb_pct_negative", "lm_neg_frac"])
    r = d["fb_pct_negative"].corr(d["lm_neg_frac"])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(d["lm_neg_frac"], d["fb_pct_negative"], s=18, alpha=0.5, color=FB)
    ax.set_xlabel("Loughran-McDonald negative-word fraction")
    ax.set_ylabel("FinBERT share of sentences negative")
    ax.set_title(f"Two ways to measure risk-factor negativity\nfiling-level correlation r = {r:.2f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tercile_bars(sorts: pd.DataFrame, path: Path) -> None:
    outcomes = ["aret_1m", "aret_3m", "aret_6m", "fwd_vol"]
    tones = sorts["tone"].unique()
    fig, axes = plt.subplots(len(tones), len(outcomes),
                             figsize=(3.2 * len(outcomes), 2.8 * len(tones)),
                             squeeze=False)
    for i, tone in enumerate(tones):
        for j, oc in enumerate(outcomes):
            ax = axes[i][j]
            row = sorts[(sorts.tone == tone) & (sorts.outcome == oc)]
            if row.empty:
                ax.axis("off")
                continue
            row = row.iloc[0]
            vals = [row["T1_low"], row["T2"], row["T3_high"]]
            ax.bar(["low", "mid", "high"], vals, color=["#2a9d8f", "#8d99ae", LM])
            ax.axhline(0, color="k", lw=0.7)
            ax.set_title(f"{tone}\n{oc}  (H−L t={row['hml_t']:.1f})", fontsize=8)
            ax.tick_params(labelsize=8)
    fig.suptitle("Forward outcomes by within-year tone tercile", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_coef_forest(grid: pd.DataFrame, path: Path) -> None:
    g = grid.dropna(subset=["coef", "se"]).copy()
    g["label"] = g["tone"] + "  →  " + g["outcome"]
    g = g.sort_values(["outcome", "tone"]).reset_index(drop=True)
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(g) + 1.5))
    ax.errorbar(g["coef"], y, xerr=1.96 * g["se"], fmt="o", ms=5,
                color=FB, ecolor="#8d99ae", capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(g["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient on z(tone), per 1 SD  (firm + year FE, 2-way clustered)")
    ax.set_title("Does risk-factor tone predict forward outcomes?")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
