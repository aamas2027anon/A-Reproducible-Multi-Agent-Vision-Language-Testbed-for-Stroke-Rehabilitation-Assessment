from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config import CLASS_NAMES, NUM_PRIMITIVE_CLASSES, portable_path


COUNT_VIDEO_KEYS = ("subject_id", "task_name", "clip_id")
COUNT_METADATA_COLUMNS = ("run_name", "fold_id", "split")
COUNT_METRIC_COLUMNS = (
    "aggregation",
    "label",
    "run_name",
    "split",
    "n_subjects",
    "n_percentage_valid",
    "actual_mean",
    "actual_sd",
    "predicted_mean",
    "predicted_sd",
    "mean_predicted_percent_of_actual",
    "mean_signed_percent_error",
    "sd_signed_percent_error",
    "mean_absolute_percent_error",
    "mae_count",
    "rmse_count",
    "pearson_r",
    "pearson_p",
)


def collapse_events(
    frame: pd.DataFrame, label_column: str, expected_step: int
) -> list[dict[str, int]]:
    if frame.empty:
        return []
    ordered = frame.sort_values("absolute_frame_index")
    first = ordered.iloc[0]
    start = previous = int(first["absolute_frame_index"])
    label = int(first[label_column])
    events: list[dict[str, int]] = []
    for row in ordered.iloc[1:].itertuples(index=False):
        current = int(row.absolute_frame_index)
        current_label = int(getattr(row, label_column))
        if current - previous != expected_step or current_label != label:
            events.append({"label_id": label, "start": start, "end": previous})
            start, label = current, current_label
        previous = current
    events.append({"label_id": label, "start": start, "end": previous})
    return events


def add_count_errors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    actual = result["actual_count"].to_numpy(dtype=float)
    predicted = result["predicted_count"].to_numpy(dtype=float)
    error = predicted - actual
    valid = actual != 0
    predicted_percent = np.full(len(result), np.nan, dtype=float)
    predicted_percent[valid] = 100.0 * predicted[valid] / actual[valid]
    result["count_error"] = error
    result["absolute_count_error"] = np.abs(error)
    result["predicted_percent_of_actual"] = predicted_percent
    result["signed_percent_error"] = predicted_percent - 100.0
    result["absolute_percent_error"] = np.abs(predicted_percent - 100.0)
    return result


def count_video_primitives(primitive_frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *COUNT_VIDEO_KEYS,
        *COUNT_METADATA_COLUMNS,
        "absolute_frame_index",
        "actual_label_id",
        "predicted_label_id",
    }
    missing = required - set(primitive_frame)
    if missing:
        raise KeyError(f"Counting frame is missing columns: {sorted(missing)}")
    if primitive_frame.empty:
        raise ValueError("Cannot count primitives from an empty prediction frame")
    timeline_keys = [*COUNT_VIDEO_KEYS, "absolute_frame_index"]
    if primitive_frame.duplicated(timeline_keys).any():
        raise ValueError("Counting requires one reconstructed prediction per video frame")

    rows: list[dict[str, Any]] = []
    for video_keys, video in primitive_frame.groupby(list(COUNT_VIDEO_KEYS), sort=True):
        ordered = video.sort_values("absolute_frame_index")
        for column in COUNT_METADATA_COLUMNS:
            if ordered[column].nunique(dropna=False) != 1:
                raise ValueError(f"{column} changes within video {video_keys}")
        actual_events = collapse_events(ordered, "actual_label_id", expected_step=1)
        predicted_events = collapse_events(ordered, "predicted_label_id", expected_step=1)
        actual_counts = np.bincount(
            [event["label_id"] for event in actual_events],
            minlength=NUM_PRIMITIVE_CLASSES,
        )
        predicted_counts = np.bincount(
            [event["label_id"] for event in predicted_events],
            minlength=NUM_PRIMITIVE_CLASSES,
        )
        first = ordered.iloc[0]
        for label_id, primitive in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "subject_id": video_keys[0],
                    "task_name": video_keys[1],
                    "clip_id": video_keys[2],
                    "label_id": label_id,
                    "primitive": primitive,
                    "actual_count": int(actual_counts[label_id]),
                    "predicted_count": int(predicted_counts[label_id]),
                    "observed_frames": int(len(ordered)),
                    "first_frame": int(ordered["absolute_frame_index"].iloc[0]),
                    "last_frame": int(ordered["absolute_frame_index"].iloc[-1]),
                    "run_name": str(first["run_name"]),
                    "fold_id": int(first["fold_id"]),
                    "split": str(first["split"]),
                }
            )
    result = add_count_errors(pd.DataFrame(rows))
    expected_rows = (
        primitive_frame.groupby(list(COUNT_VIDEO_KEYS)).ngroups
        * NUM_PRIMITIVE_CLASSES
    )
    if len(result) != expected_rows:
        raise AssertionError("Every video must contain one count row per primitive class")
    return result


def aggregate_count_pairs(
    video_counts: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    keys = [*group_columns, *COUNT_METADATA_COLUMNS]
    result = (
        video_counts.groupby(keys, as_index=False, sort=True)
        .agg(
            actual_count=("actual_count", "sum"),
            predicted_count=("predicted_count", "sum"),
        )
    )
    result["actual_count"] = result["actual_count"].astype(int)
    result["predicted_count"] = result["predicted_count"].astype(int)
    return add_count_errors(result)


def build_count_aggregations(video_counts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "video_primitive": video_counts,
        "subject_primitive": aggregate_count_pairs(
            video_counts, ("subject_id", "label_id", "primitive")
        ),
        "subject_activity": aggregate_count_pairs(
            video_counts, ("subject_id", "task_name")
        ),
        "subject_total": aggregate_count_pairs(video_counts, ("subject_id",)),
    }


def assert_count_reconciliation(aggregations: Mapping[str, pd.DataFrame]) -> None:
    reference = (
        aggregations["subject_total"][["subject_id", "actual_count", "predicted_count"]]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    for name in ("video_primitive", "subject_primitive", "subject_activity"):
        candidate = (
            aggregations[name]
            .groupby("subject_id", as_index=False)
            .agg(
                actual_count=("actual_count", "sum"),
                predicted_count=("predicted_count", "sum"),
            )
            .sort_values("subject_id")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            candidate,
            reference,
            check_dtype=False,
            obj=f"{name} counts reconciled to subject totals",
        )


def _count_pair_metrics(
    frame: pd.DataFrame, aggregation: str, label: str
) -> dict[str, Any]:
    if frame.empty or not frame["subject_id"].is_unique:
        raise ValueError(f"{aggregation}/{label} requires one row per subject")
    actual = frame["actual_count"].to_numpy(dtype=float)
    predicted = frame["predicted_count"].to_numpy(dtype=float)
    percent = frame["predicted_percent_of_actual"].to_numpy(dtype=float)
    signed_percent = frame["signed_percent_error"].to_numpy(dtype=float)
    absolute_percent = frame["absolute_percent_error"].to_numpy(dtype=float)
    valid_percent = np.isfinite(percent)
    if len(frame) >= 2 and np.unique(actual).size > 1 and np.unique(predicted).size > 1:
        correlation = pearsonr(actual, predicted)
        pearson_r, pearson_p = float(correlation.statistic), float(correlation.pvalue)
    else:
        pearson_r = pearson_p = math.nan
    return {
        "aggregation": aggregation,
        "label": label,
        "run_name": str(frame["run_name"].iloc[0]),
        "split": str(frame["split"].iloc[0]),
        "n_subjects": int(len(frame)),
        "n_percentage_valid": int(valid_percent.sum()),
        "actual_mean": float(actual.mean()),
        "actual_sd": float(actual.std(ddof=1)) if len(actual) > 1 else math.nan,
        "predicted_mean": float(predicted.mean()),
        "predicted_sd": float(predicted.std(ddof=1)) if len(predicted) > 1 else math.nan,
        "mean_predicted_percent_of_actual": (
            float(np.nanmean(percent)) if valid_percent.any() else math.nan
        ),
        "mean_signed_percent_error": (
            float(np.nanmean(signed_percent)) if valid_percent.any() else math.nan
        ),
        "sd_signed_percent_error": (
            float(np.nanstd(signed_percent, ddof=1))
            if valid_percent.sum() > 1
            else math.nan
        ),
        "mean_absolute_percent_error": (
            float(np.nanmean(absolute_percent)) if valid_percent.any() else math.nan
        ),
        "mae_count": float(np.mean(np.abs(predicted - actual))),
        "rmse_count": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }


def summarize_count_pairs(
    frame: pd.DataFrame, aggregation: str, label_column: str | None
) -> pd.DataFrame:
    groups = [("overall", frame)] if label_column is None else list(
        frame.groupby(label_column, sort=True)
    )
    rows = [
        _count_pair_metrics(group, aggregation, str(label))
        for label, group in groups
    ]
    return pd.DataFrame(rows)[list(COUNT_METRIC_COLUMNS)]


def _plot_count_bars(summary: pd.DataFrame, output: Path, title: str) -> None:
    x = np.arange(len(summary))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(8.0, 1.05 * len(summary)), 6.0))
    axis.bar(
        x - width / 2,
        summary["actual_mean"],
        width,
        yerr=summary["actual_sd"],
        capsize=3,
        label="Actual",
        color="#4c78a8",
    )
    axis.bar(
        x + width / 2,
        summary["predicted_mean"],
        width,
        yerr=summary["predicted_sd"],
        capsize=3,
        label="Predicted",
        color="#f58518",
    )
    axis.set_xticks(x, summary["label"], rotation=35, ha="right")
    axis.set(
        xlabel=summary["aggregation"].iloc[0].replace("_", " "),
        ylabel="Primitive event count",
        title=title,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_overall_count_scatter(frame: pd.DataFrame, output: Path, title: str) -> None:
    actual = frame["actual_count"].to_numpy(dtype=float)
    predicted = frame["predicted_count"].to_numpy(dtype=float)
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    padding = max(1.0, 0.05 * max(upper - lower, 1.0))
    figure, axis = plt.subplots(figsize=(7.5, 7.0))
    axis.scatter(actual, predicted, s=55, alpha=0.8)
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        "k--",
    )
    axis.set(
        xlim=(lower - padding, upper + padding),
        ylim=(lower - padding, upper + padding),
        xlabel="Actual total primitive count",
        ylabel="Predicted total primitive count",
        title=title,
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def stringify_path_tree(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, Mapping):
        return {key: stringify_path_tree(item) for key, item in value.items()}
    return value


def generate_counting_artifacts(
    primitive_frame: pd.DataFrame,
    run_root: Path,
    run_name: str,
    split: str,
) -> dict[str, Any]:
    counting_root = Path(run_root) / "counting"
    directories = {
        name: counting_root / name for name in ("predictions", "tables", "plots")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    video_counts = count_video_primitives(primitive_frame)
    aggregations = build_count_aggregations(video_counts)
    assert_count_reconciliation(aggregations)
    prediction_paths = {
        key: directories["predictions"] / f"{split}_{key}_counts.csv"
        for key in aggregations
    }
    for key, path in prediction_paths.items():
        aggregations[key].to_csv(path, index=False)

    summaries = {
        "by_primitive": summarize_count_pairs(
            aggregations["subject_primitive"], "primitive", "primitive"
        ),
        "by_activity": summarize_count_pairs(
            aggregations["subject_activity"], "activity", "task_name"
        ),
        "overall": summarize_count_pairs(
            aggregations["subject_total"], "overall", None
        ),
    }
    table_paths = {
        key: directories["tables"] / f"{split}_count_metrics_{key}.csv"
        for key in summaries
    }
    for key, path in table_paths.items():
        summaries[key].to_csv(path, index=False)

    plot_paths = {
        "by_primitive": directories["plots"] / f"{split}_counts_by_primitive.png",
        "by_activity": directories["plots"] / f"{split}_counts_by_activity.png",
        "overall": directories["plots"] / f"{split}_overall_count_actual_vs_predicted.png",
    }
    _plot_count_bars(
        summaries["by_primitive"],
        plot_paths["by_primitive"],
        f"{run_name.replace('_', ' ').title()}: counts by primitive",
    )
    _plot_count_bars(
        summaries["by_activity"],
        plot_paths["by_activity"],
        f"{run_name.replace('_', ' ').title()}: counts by activity",
    )
    _plot_overall_count_scatter(
        aggregations["subject_total"],
        plot_paths["overall"],
        f"{run_name.replace('_', ' ').title()}: total primitive counts per subject",
    )

    artifacts: dict[str, Any] = {
        "root": counting_root,
        "predictions": prediction_paths,
        "tables": table_paths,
        "plots": plot_paths,
    }
    manifest_path = counting_root / "counting_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "counting_complete": True,
                "run_name": run_name,
                "split": split,
                "subject_count": int(primitive_frame["subject_id"].nunique()),
                "method": "run_length_collapse_complete_five_horizon_timeline",
                "smoothing": "none",
                "artifacts": stringify_path_tree(artifacts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts["manifest"] = manifest_path
    return artifacts
