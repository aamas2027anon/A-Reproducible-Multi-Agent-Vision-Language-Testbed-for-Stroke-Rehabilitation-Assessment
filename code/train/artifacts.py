from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from config import (
    CLASS_NAMES,
    EVENT_IOU_THRESHOLD,
    EXPECTED_TRAIN_SUBJECTS,
    FRAME_SKIP,
    LABEL_VOCAB,
    NUM_PRIMITIVE_CLASSES,
    POSITION_NAMES,
    RuntimeConfig,
    TARGET_WINDOW_SIZE,
    TASKS,
    UEFMA_MAX,
    UEFMA_MIN,
    VARIANTS,
    portable_path,
)
from counting import generate_counting_artifacts, stringify_path_tree


ALIGN_COLUMNS = [
    "subject_id",
    "clip_id",
    "task_name",
    "target_frame_index",
    "horizon",
    "absolute_frame_index",
]
PRIMITIVE_ALIGN_COLUMNS = [
    "subject_id",
    "clip_id",
    "task_name",
    "target_frame_index",
    "horizon",
]


def aggregate_oof_predictions(config: RuntimeConfig) -> dict[str, dict[str, Path]]:
    pooled_root = config.output_root / "oof"
    assignment_path = config.output_root / "fold_assignments.csv"
    if not assignment_path.is_file():
        raise FileNotFoundError(assignment_path)
    assignments = pd.read_csv(assignment_path)[["subject_id", "fold_id"]]
    assignments = assignments.sort_values("subject_id").reset_index(drop=True)
    if len(assignments) != EXPECTED_TRAIN_SUBJECTS or not assignments["subject_id"].is_unique:
        raise AssertionError("Persisted assignments must contain one row per held-out subject")
    fold_sizes = assignments.groupby("fold_id").size().sort_index()
    fold_ids = tuple(int(value) for value in fold_sizes.index)
    if len(fold_ids) != config.n_folds:
        raise AssertionError("Persisted assignments do not match configured fold count")
    results: dict[str, dict[str, Path]] = {}
    reference_truth: pd.DataFrame | None = None
    for run_name, _ in VARIANTS:
        primitive_frames = []
        fma_frames = []
        combined_frames = []
        for fold_id in fold_ids:
            prediction_dir = (
                config.output_root
                / "folds"
                / f"fold_{fold_id:02d}"
                / run_name
                / "predictions"
            )
            required = {
                "primitive": prediction_dir / "oof_primitive_predictions.csv",
                "ue_fma": prediction_dir / "oof_ue_fma_subject_predictions.csv",
                "combined": prediction_dir / "oof_combined_window_predictions.csv",
            }
            for path in required.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            primitive_frames.append(pd.read_csv(required["primitive"]))
            fma_frames.append(pd.read_csv(required["ue_fma"]))
            combined_frames.append(pd.read_csv(required["combined"]))
        primitive = pd.concat(primitive_frames, ignore_index=True).sort_values(
            PRIMITIVE_ALIGN_COLUMNS
        ).reset_index(drop=True)
        fma = pd.concat(fma_frames, ignore_index=True).sort_values("subject_id").reset_index(
            drop=True
        )
        combined = pd.concat(combined_frames, ignore_index=True).sort_values(
            ALIGN_COLUMNS
        ).reset_index(drop=True)
        if fma["subject_id"].nunique() != EXPECTED_TRAIN_SUBJECTS:
            raise AssertionError(f"{run_name} does not have 32 OOF UE-FMA predictions")
        if not fma["subject_id"].is_unique:
            raise AssertionError(f"{run_name} contains duplicate OOF subjects")
        if set(fma["split"]) != {"oof"} or set(primitive["split"]) != {"oof"}:
            raise AssertionError("OOF rows must have split='oof'")
        if set(fma["fold_id"]) != set(fold_ids):
            raise AssertionError("OOF UE-FMA rows do not cover all folds")
        observed_assignments = fma[["subject_id", "fold_id"]].sort_values(
            "subject_id"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            observed_assignments,
            assignments,
            check_dtype=False,
            obj=f"{run_name} OOF fold assignments",
        )
        expected_subjects = set(assignments["subject_id"])
        if set(primitive["subject_id"]) != expected_subjects:
            raise AssertionError(f"{run_name} primitive predictions do not cover all subjects")
        if set(combined["subject_id"]) != expected_subjects:
            raise AssertionError(f"{run_name} combined predictions do not cover all subjects")
        if combined.duplicated(ALIGN_COLUMNS).any():
            raise AssertionError(f"Duplicate OOF primitive keys for {run_name}")
        truth = combined[[*ALIGN_COLUMNS, "actual_primitive", "actual_ue_fma"]]
        if reference_truth is None:
            reference_truth = truth
        elif not truth.equals(reference_truth):
            raise AssertionError("Base and prefix OOF ground truth are not aligned")

        prediction_dir = pooled_root / run_name / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "primitive": prediction_dir / "oof_primitive_predictions.csv",
            "ue_fma": prediction_dir / "oof_ue_fma_subject_predictions.csv",
            "combined": prediction_dir / "oof_combined_window_predictions.csv",
        }
        primitive.to_csv(paths["primitive"], index=False)
        fma.to_csv(paths["ue_fma"], index=False)
        combined.to_csv(paths["combined"], index=False)
        results[run_name] = paths

    manifest = {
        "aggregation_complete": True,
        "subject_count": EXPECTED_TRAIN_SUBJECTS,
        "fold_count": len(fold_ids),
        "fold_sizes": {
            str(int(fold_id)): int(size) for fold_id, size in fold_sizes.items()
        },
        "variants": {
            name: {key: portable_path(path) for key, path in paths.items()}
            for name, paths in results.items()
        },
    }
    pooled_root.mkdir(parents=True, exist_ok=True)
    (pooled_root / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return results


def classification_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Cannot calculate primitive metrics from an empty frame")
    actual = frame["actual_label_id"].to_numpy(dtype=int)
    predicted = frame["predicted_label_id"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(range(NUM_PRIMITIVE_CLASSES)),
        zero_division=0,
    )
    reach_id, reposition_id = LABEL_VOCAB["reach"], LABEL_VOCAB["reposition"]
    return {
        "support": int(support.sum()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=list(range(NUM_PRIMITIVE_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=list(range(NUM_PRIMITIVE_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        ),
        "reach_precision": float(precision[reach_id]),
        "reach_recall": float(recall[reach_id]),
        "reach_f1": float(f1[reach_id]),
        "reposition_precision": float(precision[reposition_id]),
        "reposition_recall": float(recall[reposition_id]),
        "reposition_f1": float(f1[reposition_id]),
    }


def per_class_summary(frame: pd.DataFrame) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        frame["actual_label_id"],
        frame["predicted_label_id"],
        labels=list(range(NUM_PRIMITIVE_CLASSES)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "label_id": range(NUM_PRIMITIVE_CLASSES),
            "label": CLASS_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
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


def temporal_iou(first: Mapping[str, int], second: Mapping[str, int], sample_step: int) -> float:
    intersection = max(
        0,
        min(first["end"], second["end"])
        - max(first["start"], second["start"])
        + sample_step,
    )
    first_length = first["end"] - first["start"] + sample_step
    second_length = second["end"] - second["start"] + sample_step
    union = first_length + second_length - intersection
    return float(intersection / union) if union else 0.0


def match_event_lists(
    actual_events: Sequence[Mapping[str, int]],
    predicted_events: Sequence[Mapping[str, int]],
    sample_step: int,
) -> list[tuple[int, int, float]]:
    candidates = []
    for actual_index, actual in enumerate(actual_events):
        for predicted_index, predicted in enumerate(predicted_events):
            if actual["label_id"] != predicted["label_id"]:
                continue
            iou = temporal_iou(actual, predicted, sample_step)
            if iou >= EVENT_IOU_THRESHOLD:
                candidates.append((iou, actual_index, predicted_index))
    used_actual: set[int] = set()
    used_predicted: set[int] = set()
    matches = []
    for iou, actual_index, predicted_index in sorted(candidates, reverse=True):
        if actual_index in used_actual or predicted_index in used_predicted:
            continue
        used_actual.add(actual_index)
        used_predicted.add(predicted_index)
        matches.append((actual_index, predicted_index, iou))
    return matches


def event_metrics(frame: pd.DataFrame, expected_step: int = FRAME_SKIP) -> pd.DataFrame:
    counts = {
        label_id: {"tp": 0, "fp": 0, "fn": 0}
        for label_id in range(NUM_PRIMITIVE_CLASSES)
    }
    for _, clip in frame.groupby(["subject_id", "task_name", "clip_id"], sort=False):
        actual = collapse_events(clip, "actual_label_id", expected_step)
        predicted = collapse_events(clip, "predicted_label_id", expected_step)
        matches = match_event_lists(actual, predicted, expected_step)
        matched_actual = {match[0] for match in matches}
        matched_predicted = {match[1] for match in matches}
        for index, event in enumerate(actual):
            counts[event["label_id"]]["tp" if index in matched_actual else "fn"] += 1
        for index, event in enumerate(predicted):
            if index not in matched_predicted:
                counts[event["label_id"]]["fp"] += 1
    rows = []
    for label_id, label in enumerate(CLASS_NAMES):
        tp, fp, fn = (counts[label_id][key] for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label_id": label_id,
                "label": label,
                "iou_threshold": EVENT_IOU_THRESHOLD,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn,
            }
        )
    result = pd.DataFrame(rows)
    totals = result[["tp", "fp", "fn"]].sum()
    precision = totals.tp / (totals.tp + totals.fp) if totals.tp + totals.fp else 0.0
    recall = totals.tp / (totals.tp + totals.fn) if totals.tp + totals.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result.loc[len(result)] = {
        "label_id": -1,
        "label": "overall",
        "iou_threshold": EVENT_IOU_THRESHOLD,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(totals.tp),
        "fp": int(totals.fp),
        "fn": int(totals.fn),
        "support": int(totals.tp + totals.fn),
    }
    return result


def save_confusion_matrices(frame: pd.DataFrame, output_dir: Path, run_name: str) -> None:
    """Per-horizon confusion matrices, in raw counts and row percentages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for horizon, position_name in enumerate(POSITION_NAMES):
        selected = frame.query("horizon == @horizon")
        matrix = confusion_matrix(
            selected["actual_label_id"],
            selected["predicted_label_id"],
            labels=list(range(NUM_PRIMITIVE_CLASSES)),
        )
        totals = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(
            matrix,
            totals,
            out=np.zeros_like(matrix, dtype=float),
            where=totals != 0,
        ) * 100
        for values, suffix in ((matrix, "counts"), (percentages, "row_percent")):
            figure, axis = plt.subplots(figsize=(7.5, 6.5))
            image = axis.imshow(values, cmap="Blues")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.set_xticks(range(NUM_PRIMITIVE_CLASSES), CLASS_NAMES, rotation=45, ha="right")
            axis.set_yticks(range(NUM_PRIMITIVE_CLASSES), CLASS_NAMES)
            axis.set(
                xlabel="Predicted primitive",
                ylabel="Actual primitive",
                title=f"{run_name.replace('_', ' ').title()} — {position_name} ({suffix})",
            )
            threshold = values.max() / 2 if values.size else 0.0
            for row in range(NUM_PRIMITIVE_CLASSES):
                for column in range(NUM_PRIMITIVE_CLASSES):
                    value = values[row, column]
                    label = f"{int(value)}" if suffix == "counts" else f"{value:.1f}%"
                    axis.text(
                        column,
                        row,
                        label,
                        ha="center",
                        va="center",
                        color="white" if value > threshold else "black",
                    )
            figure.tight_layout()
            figure.savefig(
                output_dir / f"confusion_{position_name}_{suffix}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(figure)


def _score_clips(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, clip in frame.groupby(["subject_id", "task_name", "clip_id"], sort=False):
        ordered = clip.sort_values("absolute_frame_index")
        if ordered["absolute_frame_index"].duplicated().any():
            raise ValueError(f"Duplicate reconstructed frames in clip {keys}")
        rows.append(
            {
                "subject_id": keys[0],
                "task_name": keys[1],
                "clip_id": keys[2],
                "support": len(ordered),
                "accuracy": accuracy_score(
                    ordered["actual_label_id"], ordered["predicted_label_id"]
                ),
                "weighted_f1": f1_score(
                    ordered["actual_label_id"],
                    ordered["predicted_label_id"],
                    labels=list(range(NUM_PRIMITIVE_CLASSES)),
                    average="weighted",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_timeline(
    frame: pd.DataFrame, row: Any, quality: str, run_name: str, output_dir: Path
) -> None:
    selected = frame[
        (frame["subject_id"] == row.subject_id)
        & (frame["task_name"] == row.task_name)
        & (frame["clip_id"] == row.clip_id)
    ].sort_values("absolute_frame_index")
    frames = selected["absolute_frame_index"].to_numpy()
    cmap = plt.get_cmap("tab10", NUM_PRIMITIVE_CLASSES)
    figure, axes = plt.subplots(2, 1, figsize=(16, 4.2), sharex=True)
    for axis, values, label in zip(
        axes,
        (
            selected["actual_label_id"].to_numpy(),
            selected["predicted_label_id"].to_numpy(),
        ),
        ("Actual", "Predicted"),
    ):
        axis.imshow(
            values[None, :],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=NUM_PRIMITIVE_CLASSES - 1,
            extent=[frames.min(), frames.max() + 1, 0, 1],
            interpolation="nearest",
        )
        axis.set_yticks([])
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Frame index")
    figure.suptitle(
        f"{quality.title()} {row.task_name} | {row.subject_id}, {row.clip_id}, {run_name} | "
        f"weighted F1={row.weighted_f1:.3f}"
    )
    figure.legend(
        handles=[Patch(facecolor=cmap(index), label=label) for index, label in enumerate(CLASS_NAMES)],
        loc="lower center",
        ncol=NUM_PRIMITIVE_CLASSES,
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.91))
    safe_clip = str(row.clip_id).replace("/", "_")
    figure.savefig(
        output_dir / f"timeline_{row.task_name}_{quality}_{row.subject_id}_{safe_clip}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def generate_pooled_diagnostics(config: RuntimeConfig) -> dict[str, dict[str, Any]]:
    counting_results: dict[str, dict[str, Any]] = {}
    for run_name, _ in VARIANTS:
        root = config.output_root / "oof" / run_name
        combined_path = root / "predictions" / "oof_combined_window_predictions.csv"
        fma_path = root / "predictions" / "oof_ue_fma_subject_predictions.csv"
        frame = pd.read_csv(combined_path)
        tables = root / "tables"
        plots = root / "plots"
        timelines = root / "timelines"
        for path in (tables, plots, timelines):
            path.mkdir(parents=True, exist_ok=True)
        save_confusion_matrices(frame, root / "matrices", run_name)

        position_rows = []
        class_frames = []
        event_frames = []
        for horizon, position in enumerate(POSITION_NAMES):
            selected = frame.query("horizon == @horizon")
            summary = classification_summary(selected)
            events = event_metrics(selected)
            position_rows.append(
                {
                    "run_name": run_name,
                    "prediction_position": position,
                    **summary,
                    "event_level_f1": float(events.query("label == 'overall'")["f1"].iloc[0]),
                }
            )
            classes = per_class_summary(selected)
            classes.insert(0, "prediction_position", position)
            classes.insert(0, "run_name", run_name)
            class_frames.append(classes)
            events.insert(0, "prediction_position", position)
            events.insert(0, "run_name", run_name)
            event_frames.append(events)
        position_table = pd.DataFrame(position_rows)
        position_table.to_csv(tables / "primitive_position_metrics.csv", index=False)
        pd.concat(class_frames, ignore_index=True).to_csv(
            tables / "primitive_per_class_metrics.csv", index=False
        )
        pd.concat(event_frames, ignore_index=True).to_csv(
            tables / "primitive_event_metrics.csv", index=False
        )

        task_dir = tables / "task_metrics"
        task_dir.mkdir(exist_ok=True)
        for task in TASKS:
            rows = []
            task_frame = frame.query("task_name == @task")
            if task_frame.empty:
                raise ValueError(f"No OOF predictions for task {task}")
            for horizon, position in enumerate(POSITION_NAMES):
                selected = task_frame.query("horizon == @horizon")
                events = event_metrics(selected)
                rows.append(
                    {
                        "task_name": task,
                        "run_name": run_name,
                        "prediction_position": position,
                        **classification_summary(selected),
                        "event_level_f1": float(
                            events.query("label == 'overall'")["f1"].iloc[0]
                        ),
                    }
                )
            pd.DataFrame(rows).to_csv(task_dir / f"{task}_metrics.csv", index=False)

        clip_scores = _score_clips(frame)
        clip_scores.to_csv(tables / "primitive_clip_metrics.csv", index=False)
        for task in TASKS:
            scores = clip_scores.query("task_name == @task")
            best = scores.sort_values(
                ["weighted_f1", "accuracy", "subject_id", "clip_id"],
                ascending=[False, False, True, True],
            ).iloc[0]
            worst = scores.sort_values(
                ["weighted_f1", "accuracy", "subject_id", "clip_id"],
                ascending=[True, True, True, True],
            ).iloc[0]
            _plot_timeline(frame, best, "best", run_name, timelines)
            _plot_timeline(frame, worst, "worst", run_name, timelines)

        figure, axis = plt.subplots(figsize=(9, 5.5))
        axis.plot(
            position_table["prediction_position"],
            position_table["macro_f1"],
            marker="o",
            linewidth=2,
        )
        axis.set(
            xlabel="Primitive prediction position",
            ylabel="Macro F1",
            title=f"{run_name.replace('_', ' ').title()}: OOF macro F1",
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots / "primitive_macro_f1_by_position.png", dpi=300)
        plt.close(figure)

        fma = pd.read_csv(fma_path)
        actual = fma["actual_ue_fma"].to_numpy(float)
        predicted = fma["predicted_ue_fma"].to_numpy(float)
        pearson = (
            float(np.corrcoef(actual, predicted)[0, 1])
            if np.unique(predicted).size > 1
            else np.nan
        )
        pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "mae": mean_absolute_error(actual, predicted),
                    "rmse": mean_squared_error(actual, predicted) ** 0.5,
                    "r2": r2_score(actual, predicted),
                    "pearson_r": pearson,
                }
            ]
        ).to_csv(tables / "ue_fma_metrics.csv", index=False)
        lower = max(UEFMA_MIN, float(np.min(np.r_[actual, predicted])) - 2)
        upper = min(UEFMA_MAX, float(np.max(np.r_[actual, predicted])) + 2)
        figure, axis = plt.subplots(figsize=(7.5, 7))
        axis.scatter(actual, predicted, s=55, alpha=0.8)
        axis.plot([lower, upper], [lower, upper], "k--")
        axis.set(
            xlim=(lower, upper),
            ylim=(lower, upper),
            xlabel="Actual UE-FMA",
            ylabel="Predicted UE-FMA",
            title=f"{run_name.replace('_', ' ').title()}: OOF UE-FMA",
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots / "ue_fma_actual_vs_predicted.png", dpi=300)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7.5, 5.5))
        bar = axis.bar([run_name], [pearson], color="#f58518")[0]
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(
            ylim=(-1.0, 1.0),
            xlabel="Run",
            ylabel="Pearson correlation coefficient (r)",
            title=f"{run_name.replace('_', ' ').title()}: OOF UE-FMA Pearson r",
        )
        label = "NaN" if pd.isna(pearson) else f"{pearson:.3f}"
        height = 0.0 if pd.isna(pearson) else pearson
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.04 if height >= 0 else -0.08),
            label,
            ha="center",
        )
        figure.tight_layout()
        figure.savefig(plots / "ue_fma_pearson.png", dpi=300)
        plt.close(figure)

        counting_results[run_name] = generate_counting_artifacts(
            frame, root, run_name, "oof"
        )

    aggregation_manifest_path = config.output_root / "oof" / "aggregation_manifest.json"
    aggregation_manifest = json.loads(aggregation_manifest_path.read_text(encoding="utf-8"))
    aggregation_manifest["counting"] = stringify_path_tree(counting_results)
    aggregation_manifest_path.write_text(
        json.dumps(aggregation_manifest, indent=2), encoding="utf-8"
    )
    return counting_results


def write_final_artifacts(
    combined: pd.DataFrame, run_dir: Path, run_name: str, split: str
) -> dict[str, Any]:
    """End-of-run metrics, kept separate from the per-epoch files in `epochs/`.

    Writes `final/` under `run_dir`: position, per-class, event and task tables, the
    confusion matrices and the counting artifacts. Used by the multi-seed stage and by
    the multi-agent evaluation, so both produce the same layout.
    """
    final_root = run_dir / "final"
    tables = final_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    save_confusion_matrices(combined, final_root / "matrices", run_name)

    position_rows = []
    class_frames = []
    event_frames = []
    for horizon, position in enumerate(POSITION_NAMES):
        selected = combined.loc[combined["horizon"] == horizon]
        events = event_metrics(selected)
        position_rows.append(
            {
                "run_name": run_name,
                "split": split,
                "prediction_position": position,
                **classification_summary(selected),
                "event_level_f1": float(events.query("label == 'overall'")["f1"].iloc[0]),
            }
        )
        classes = per_class_summary(selected)
        classes.insert(0, "prediction_position", position)
        classes.insert(0, "run_name", run_name)
        class_frames.append(classes)
        events.insert(0, "prediction_position", position)
        events.insert(0, "run_name", run_name)
        event_frames.append(events)
    pd.DataFrame(position_rows).to_csv(
        tables / "primitive_position_metrics.csv", index=False
    )
    pd.concat(class_frames, ignore_index=True).to_csv(
        tables / "primitive_per_class_metrics.csv", index=False
    )
    pd.concat(event_frames, ignore_index=True).to_csv(
        tables / "primitive_event_metrics.csv", index=False
    )

    task_rows = []
    for task in TASKS:
        task_frame = combined.loc[combined["task_name"] == task]
        if task_frame.empty:
            continue
        for horizon, position in enumerate(POSITION_NAMES):
            selected = task_frame.loc[task_frame["horizon"] == horizon]
            if selected.empty:
                continue
            events = event_metrics(selected)
            task_rows.append(
                {
                    "run_name": run_name,
                    "split": split,
                    "task_name": task,
                    "prediction_position": position,
                    **classification_summary(selected),
                    "event_level_f1": float(
                        events.query("label == 'overall'")["f1"].iloc[0]
                    ),
                }
            )
    task_metrics_path = tables / "task_metrics.csv"
    pd.DataFrame(task_rows).to_csv(task_metrics_path, index=False)

    counting = generate_counting_artifacts(combined, final_root, run_name, split)
    return {
        "final_root": portable_path(final_root),
        "position_metrics": portable_path(tables / "primitive_position_metrics.csv"),
        "per_class_metrics": portable_path(tables / "primitive_per_class_metrics.csv"),
        "event_metrics": portable_path(tables / "primitive_event_metrics.csv"),
        "task_metrics": portable_path(task_metrics_path),
        "counting": stringify_path_tree(counting),
    }
