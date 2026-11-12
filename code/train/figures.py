"""Empirical VLM figures, at two-column print width (7.00 in, 500 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

TWO_COL = 7.00
DPI = 500
FONT = 7.2
RNG_SEED = 9575
JITTER = 0.045

VARIANTS = ["Base", "Base + patient text", "Base + patient prefix"]
VCOLOR = {"Base": "#4c6ef5", "Base + patient text": "#f08c00", "Base + patient prefix": "#2f9e44"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": 7.8, "axes.labelsize": FONT,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
        "figure.titlesize": 8.2, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", dpi=DPI)
    plt.close(fig)
    return path


def strip_box(ax, groups, labels, colors) -> None:
    ax.boxplot(groups, showmeans=True, widths=0.52, patch_artist=False,
               medianprops={"color": "#333333"},
               meanprops={"marker": "^", "markerfacecolor": "#c23b33", "markeredgecolor": "#c23b33", "markersize": 3.6, "alpha": 0.7})
    ax.set_xticks(range(1, len(labels) + 1), labels)
    for i, (values, color) in enumerate(zip(groups, colors), 1):
        jitter = np.random.default_rng(RNG_SEED + i).normal(0, JITTER, len(values))
        ax.plot(i + jitter, values, "o", ms=2.8, alpha=0.72, color=color)
    ax.grid(alpha=0.22, axis="y", linewidth=0.5)


def vlm_metric_spread(seed_level: pd.DataFrame, significance: pd.DataFrame, path: Path) -> Path:
    """Per-seed classification metrics of the three variants, with p against Base."""
    style()
    specs = [("weighted_f1", "Weighted F1", "Weighted F1"), ("accuracy", "Accuracy", "Accuracy"),
             ("reach_f1", "Reach F1", "Reach F1"), ("reposition_f1", "Reposition F1", "Reposition F1")]
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 2.05), constrained_layout=True)
    for ax, (column, title, metric) in zip(axes, specs):
        groups = [seed_level.loc[seed_level.variant == v, column].to_numpy() for v in VARIANTS]
        strip_box(ax, groups, ["Base", "+text", "+prefix"], [VCOLOR[v] for v in VARIANTS])
        ax.set_title(title, pad=19)
        pvals = []
        for v in VARIANTS[1:]:
            row = significance[(significance.Comparison == f"{v} vs Base") & (significance.Metric == metric)]
            pvals.append(float(row.iloc[0]["p"]))
        ax.text(0.5, 1.015, "\n".join(f"Base {short} vs Base: p={pv:.3f}" for short, pv in zip(["+ text", "+ prefix"], pvals)),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6, color="#444444", linespacing=1.15)
        ax.tick_params(axis="x", pad=1)
    return save(fig, path)


def vlm_uefma(subject_predictions: pd.DataFrame, seed_level: pd.DataFrame, path: Path) -> Path:
    """Each test subject's UE-FMA prediction (mean and SD over seeds) against the truth."""
    style()
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.22), constrained_layout=True, sharex=True, sharey=True)
    titles = ["Base", "+ patient text", "+ patient prefix"]
    for ax, variant, title in zip(axes, VARIANTS, titles):
        d = subject_predictions[subject_predictions.variant == variant].groupby("subject_id").agg(
            actual=("actual_ue_fma", "first"),
            predicted=("predicted_ue_fma", "mean"),
            sd=("predicted_ue_fma", "std")).reset_index()
        ax.errorbar(d.actual, d.predicted, yerr=d.sd.fillna(0), fmt="o", ms=3.2, capsize=1.8,
                    color=VCOLOR[variant], ecolor=VCOLOR[variant], elinewidth=0.7,
                    label="Subject (mean ± SD across seeds)")
        slope, intercept = np.polyfit(d.actual, d.predicted, 1)
        xx = np.array([25, 65])
        ax.plot(xx, xx, "--", color="#777777", lw=0.7, label="Identity")
        ax.plot(xx, slope * xx + intercept, color="#c23b33", lw=1.0, label="Least-squares fit")
        ax.set_title(title)
        pooled = pearsonr(d.actual, d.predicted)
        r_seed = seed_level.loc[seed_level.variant == variant, "ue_fma_pearson_r"]
        ax.annotate(f"per-seed r = {r_seed.mean():.3f} ± {r_seed.std(ddof=1):.3f}\n"
                    f"seed-averaged r = {pooled.statistic:.3f}\n"
                    f"n = {len(d)} subjects, {r_seed.size} seeds",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=6.0,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
        ax.set_xlim(25, 65); ax.set_ylim(25, 65)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.set_xlabel("Actual UE-FMA")
    axes[0].set_ylabel("Predicted UE-FMA\n(mean ± SD across seeds)")
    axes[0].legend(loc="lower right", fontsize=5.6, framealpha=0.85, edgecolor="#cccccc")
    return save(fig, path)
