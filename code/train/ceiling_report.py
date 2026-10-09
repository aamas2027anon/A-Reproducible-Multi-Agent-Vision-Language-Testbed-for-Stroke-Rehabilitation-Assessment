"""Ceiling-test aggregation: curves and per-variant metrics, deliberately no statistics.

The ceiling test runs one seed, so there is nothing to pair and no sampling distribution
to estimate. This module therefore emits no p-value, confidence interval, or paired
delta — degenerate versions of those would only invite misreading. The question these
files answer is narrower and more useful: is the curve still climbing at the last epoch,
where does validation turn upward, and do the three variants separate at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import POSITION_NAMES, RuntimeConfig, VARIANTS
from cval_statistics import DISPLAY_NAMES
from multiseed import seed_run_dir, task_metrics_path
from multiseed_statistics import _load_seed_variant_frames, _seed_variant_row


CEILING_ANALYSIS_DIRNAME = "ceiling_analysis"
# Every column here is a measurement of one run. Nothing inferential belongs in this file.
# Tokens are unambiguous on purpose: a bare "ci" would also match "reach_precision".
FORBIDDEN_COLUMN_TOKENS = (
    "p-value",
    "p_value",
    "pvalue",
    "confidence interval",
    "confidence_interval",
    "permutation",
    "bootstrap",
    "wilcoxon",
    "mcnemar",
    "cohen",
    "rank-biserial",
    "paired delta",
    "paired_delta",
    "95%",
)


def ceiling_analysis_dir(config: RuntimeConfig) -> Path:
    return config.output_root / CEILING_ANALYSIS_DIRNAME


def _epoch_metrics_path(config: RuntimeConfig, seed: int, run_name: str) -> Path:
    return seed_run_dir(config, seed, run_name) / "epochs" / "epoch_metrics.csv"


def collect_epoch_metrics(config: RuntimeConfig, seed: int) -> pd.DataFrame:
    """The three variants' per-epoch tables, concatenated."""
    frames = []
    for run_name, conditioning in VARIANTS:
        path = _epoch_metrics_path(config, seed, run_name)
        if not path.is_file():
            print(f"No per-epoch metrics for {run_name} at seed {seed}; skipping")
            continue
        frame = pd.read_csv(path)
        frame["variant"] = DISPLAY_NAMES[run_name]
        frame["conditioning"] = conditioning
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["variant", "epoch"])


def _best_epoch(epochs: pd.DataFrame, run_name: str) -> dict[str, Any]:
    """Where test accuracy peaked, and whether it was still rising at the end."""
    variant = epochs.loc[epochs["variant"] == DISPLAY_NAMES[run_name]]
    accuracy = (
        variant[["epoch", "test_accuracy"]].dropna() if not variant.empty else variant
    )
    if accuracy.empty:
        return {
            "epochs_trained": 0,
            "best_epoch": None,
            "best_test_accuracy": float("nan"),
            "final_test_accuracy": float("nan"),
            "still_improving_at_last_epoch": None,
        }
    ordered = accuracy.sort_values("epoch")
    best = ordered.loc[ordered["test_accuracy"].idxmax()]
    last_epoch = int(ordered["epoch"].iloc[-1])
    final_accuracy = float(ordered["test_accuracy"].iloc[-1])
    return {
        "epochs_trained": last_epoch + 1,
        # Reported 1-based to match the plots and the "epoch N of 10" reading.
        "best_epoch": int(best["epoch"]) + 1,
        "best_test_accuracy": float(best["test_accuracy"]),
        "final_test_accuracy": final_accuracy,
        "still_improving_at_last_epoch": bool(int(best["epoch"]) == last_epoch),
    }


def collect_variant_metrics(
    config: RuntimeConfig, seed: int, epochs: pd.DataFrame
) -> pd.DataFrame:
    frames = _load_seed_variant_frames(config, seed)
    if frames is None:
        raise RuntimeError(
            f"Seed {seed} does not have all variants complete; nothing to report"
        )
    rows = []
    for run_name, conditioning in VARIANTS:
        row = _seed_variant_row(config, seed, run_name, conditioning, frames[run_name])
        row["n_seeds"] = 1
        row.update(_best_epoch(epochs, run_name))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_task_metrics(config: RuntimeConfig, seed: int) -> pd.DataFrame:
    frames = []
    for run_name, conditioning in VARIANTS:
        path = task_metrics_path(seed_run_dir(config, seed, run_name))
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frame["conditioning"] = conditioning
        frame["variant"] = DISPLAY_NAMES[run_name]
        frame["n_seeds"] = 1
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def export_ceiling_plots(epochs: pd.DataFrame, plot_dir: Path) -> list[Path]:
    """Three-variant overlays. These are the point of the ceiling test."""
    if epochs.empty:
        return []
    written: list[Path] = []

    for column, label, filename in (
        ("test_accuracy", "Test accuracy", "test_accuracy_by_epoch.png"),
        ("test_joint_loss", "Test joint loss", "test_loss_by_epoch.png"),
    ):
        if column not in epochs:
            continue
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        for variant, group in epochs.groupby("variant", sort=True):
            series = group[["epoch", column]].dropna().sort_values("epoch")
            axis.plot(series["epoch"] + 1, series[column], marker="o", label=variant)
        axis.set(xlabel="Epoch", ylabel=label, title=f"{label} by epoch (one seed)")
        axis.grid(alpha=0.25)
        axis.legend()
        path = plot_dir / filename
        _save(figure, path)
        written.append(path)

    # Train vs validation per variant: where these two separate is the overfitting point,
    # which with a constant learning rate is the number to read off.
    if {"train_joint_loss", "val_joint_loss"}.issubset(epochs.columns):
        variants = sorted(epochs["variant"].unique())
        figure, axes = plt.subplots(
            1, len(variants), figsize=(5.5 * len(variants), 4.8), squeeze=False, sharey=True
        )
        for axis, variant in zip(axes[0], variants):
            group = epochs.loc[epochs["variant"] == variant].sort_values("epoch")
            for column, color, label in (
                ("train_joint_loss", "#2f6b8a", "train"),
                ("val_joint_loss", "#c05a2b", "validation"),
            ):
                series = group[["epoch", column]].dropna()
                axis.plot(
                    series["epoch"] + 1, series[column], marker="o", color=color, label=label
                )
            axis.set(xlabel="Epoch", ylabel="Joint loss", title=variant)
            axis.grid(alpha=0.25)
            axis.legend()
        path = plot_dir / "train_vs_val_loss_by_variant.png"
        _save(figure, path)
        written.append(path)
    return written


def export_task_accuracy_plot(task_epochs: pd.DataFrame, plot_dir: Path) -> Path | None:
    """Per-task accuracy over epochs, one panel per variant."""
    if task_epochs.empty or "task_name" not in task_epochs:
        return None
    current = task_epochs.loc[
        task_epochs["prediction_position"] == POSITION_NAMES[0]
    ]
    if current.empty:
        return None
    variants = sorted(current["variant"].unique())
    figure, axes = plt.subplots(
        1, len(variants), figsize=(5.5 * len(variants), 4.8), squeeze=False, sharey=True
    )
    for axis, variant in zip(axes[0], variants):
        group = current.loc[current["variant"] == variant]
        for task, task_frame in group.groupby("task_name", sort=True):
            series = task_frame.sort_values("epoch")
            axis.plot(series["epoch"] + 1, series["accuracy"], marker="o", label=task)
        axis.set(xlabel="Epoch", ylabel="Accuracy (current position)", title=variant)
        axis.grid(alpha=0.25)
    axes[0][-1].legend(ncol=2, fontsize=7)
    path = plot_dir / "task_accuracy_by_epoch.png"
    _save(figure, path)
    return path


def _assert_no_inferential_columns(frame: pd.DataFrame, name: str) -> None:
    """Keep the no-statistics promise enforceable rather than aspirational."""
    offenders = [
        column
        for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN_COLUMN_TOKENS)
    ]
    if offenders:
        raise AssertionError(
            f"{name} carries inferential columns {offenders}; the ceiling test has one "
            "seed and must not report statistics"
        )


def export_ceiling_report(
    config: RuntimeConfig, seeds: Sequence[int] | None = None
) -> dict[str, Path]:
    pool = tuple(seeds) if seeds is not None else config.stage_spec.seeds
    if len(pool) != 1:
        raise ValueError(
            f"The ceiling report expects exactly one seed; got {list(pool)}"
        )
    seed = int(pool[0])

    analysis_dir = ceiling_analysis_dir(config)
    plot_dir = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs = collect_epoch_metrics(config, seed)
    variant_metrics = collect_variant_metrics(config, seed, epochs)
    task_metrics = collect_task_metrics(config, seed)
    epoch_task_metrics = _collect_epoch_task_metrics(config, seed)

    for frame, name in (
        (variant_metrics, "ceiling_variant_metrics"),
        (task_metrics, "ceiling_task_metrics"),
    ):
        _assert_no_inferential_columns(frame, name)

    outputs = {
        "variant_metrics": analysis_dir / "ceiling_variant_metrics.csv",
        "epoch_metrics": analysis_dir / "ceiling_epoch_metrics.csv",
        "task_metrics": analysis_dir / "ceiling_task_metrics.csv",
    }
    variant_metrics.to_csv(outputs["variant_metrics"], index=False, float_format="%.6f")
    epochs.to_csv(outputs["epoch_metrics"], index=False, float_format="%.6f")
    task_metrics.to_csv(outputs["task_metrics"], index=False, float_format="%.6f")

    export_ceiling_plots(epochs, plot_dir)
    export_task_accuracy_plot(epoch_task_metrics, plot_dir)
    outputs["plots"] = plot_dir

    still_climbing = variant_metrics["still_improving_at_last_epoch"].fillna(False)
    if bool(still_climbing.any()):
        climbing = variant_metrics.loc[still_climbing, "variant"].tolist()
        print(
            f"Test accuracy peaked at the last trained epoch for {climbing}; "
            "consider rerunning this stage with a higher --epochs before the sweep"
        )
    return outputs


def _collect_epoch_task_metrics(config: RuntimeConfig, seed: int) -> pd.DataFrame:
    """Per-epoch, per-task metrics written by the training callback."""
    frames = []
    for run_name, _ in VARIANTS:
        path = seed_run_dir(config, seed, run_name) / "epochs" / "epoch_task_metrics.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame["variant"] = DISPLAY_NAMES[run_name]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
