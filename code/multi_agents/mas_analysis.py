"""What the agents changed relative to the single VLM, for one MAS mode.

The VLM and every MAS mode replay the same cached model outputs, so each frame's
model cue is identical across modes and every difference is attributable to an
agent. Each frame passes through two stages:

    model cue ──planning filter──► plan ──safety revision / coordinator──► final cue
                                                                    (released or withheld)

`change_attribution.csv` scores each stage on the frames it changed: accuracy before
and after, and the net number of frames it made correct (after − before). Withheld
frames (escalate, block) keep their planned cue, as in `metrics.py`, so accuracy is
comparable with the VLM on every frame.

Outputs (in `<mode>/mas_analysis/`):
  change_attribution.csv      frames changed, accuracy before/after, net correct, per stage
  revision_outcomes.csv       safety revisions: planned -> final cue, and what was true
  class_changes.csv           per-class precision/recall/F1 and predicted share, VLM vs mode
  count_changes_by_subject.csv  per subject and primitive: actual, VLM and MAS counts
  subject_deltas.csv          per-subject accuracy, macro-F1 and unsafe-rate deltas
  headline.csv                the mode's summary metrics next to the VLM's, with deltas
  change_attribution.png, class_f1_delta.png, count_error_vs_vlm.png,
  subject_accuracy_delta.png, safety_tradeoff.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multi_agents import plots
from multi_agents.settings import ACTIVE_IDS, HORIZONS, PRIMITIVES, REST_ID, VERDICTS
from multi_agents.train_bridge import build_count_aggregations, count_video_primitives, per_class_summary

FRAME_KEYS = ["subject_id", "clip_id", "window_index", "horizon"]
COUNT_COLUMNS = [
    "subject_id",
    "clip_id",
    "task_name",
    "target_frame_index",
    "horizon",
    "actual_primitive",
    "predicted_primitive",
    "prediction_probability",
    "run_name",
    "fold_id",
    "split",
    "actual_label_id",
    "predicted_label_id",
    "absolute_frame_index",
]
HEADLINE_METRICS = (
    "accuracy_all",
    "macro_f1_all",
    "accuracy_future_primitive",
    "macro_f1_future_primitive",
    "accuracy_current",
    "reach_f1_all",
    "reposition_f1_all",
    "count_mae",
    "count_pearson_r",
    "ue_fma_subject_mae",
    "ue_fma_exercise_mae",
    "unsafe_recommendation_rate",
    "hazard_cue_rate",
    "implausible_transition_rate",
    "coverage_frames",
    "selective_accuracy",
    "unnecessary_escalation_rate",
)


def stage_frame(records: pd.DataFrame, primitives: pd.DataFrame) -> pd.DataFrame:
    """One row per scored frame: the cue after every stage, the status and the truth."""
    columns = ["model_action", "plan_action", "final_action"]
    rows = []
    for record in records[["subject_id", "clip_id", "window_index", "status", *columns]].itertuples(index=False):
        for horizon in range(HORIZONS):
            rows.append(
                (
                    record.subject_id,
                    record.clip_id,
                    int(record.window_index),
                    horizon,
                    record.status,
                    record.model_action[horizon],
                    record.plan_action[horizon],
                    record.final_action[horizon],
                )
            )
    stages = pd.DataFrame(rows, columns=[*FRAME_KEYS, "status", "model", "plan", "final"])
    truth = primitives[[*FRAME_KEYS, "actual_primitive", "released"]]
    return stages.merge(truth, on=FRAME_KEYS, how="inner", validate="one_to_one")


def change_attribution(stages: pd.DataFrame) -> pd.DataFrame:
    truth = stages["actual_primitive"]
    comparisons = [
        ("planning filter", "model", "plan", stages["plan"] != stages["model"]),
        ("safety revision / coordinator", "plan", "final", stages["final"] != stages["plan"]),
        *(
            (f"all agents, status={status}", "model", "final", (stages["status"] == status) & (stages["final"] != stages["model"]))
            for status in VERDICTS
        ),
        ("all agents", "model", "final", stages["final"] != stages["model"]),
    ]
    rows = []
    for stage, before, after, mask in comparisons:
        selected = stages[mask]
        correct_before = int((selected[before] == truth[mask]).sum())
        correct_after = int((selected[after] == truth[mask]).sum())
        rows.append(
            {
                "stage": stage,
                "compared": f"{before} -> {after}",
                "frames_changed": int(mask.sum()),
                "share_of_frames": float(mask.mean()),
                "accuracy_before": correct_before / len(selected) if len(selected) else np.nan,
                "accuracy_after": correct_after / len(selected) if len(selected) else np.nan,
                "correct_before": correct_before,
                "correct_after": correct_after,
                "net_correct_frames": correct_after - correct_before,
            }
        )
    return pd.DataFrame(rows)


def revision_outcomes(stages: pd.DataFrame) -> pd.DataFrame:
    revised = stages[(stages["status"] == "revise") & (stages["final"] != stages["plan"])]
    rows = []
    for (planned, final), group in revised.groupby(["plan", "final"]):
        truth = group["actual_primitive"]
        rows.append(
            {
                "planned": planned,
                "final": final,
                "frames": len(group),
                "final_was_true": float((truth == final).mean()),
                "planned_was_true": float((truth == planned).mean()),
                "neither_was_true": float(((truth != final) & (truth != planned)).mean()),
                **{f"true_{name}": float((truth == name).mean()) for name in PRIMITIVES},
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("frames", ascending=False).reset_index(drop=True) if len(frame) else frame


def class_changes(primitives: pd.DataFrame, vlm_primitives: pd.DataFrame) -> pd.DataFrame:
    base = per_class_summary(vlm_primitives).set_index("label")
    mode = per_class_summary(primitives).set_index("label")
    frame = pd.DataFrame({"support": base["support"]})
    for metric in ("precision", "recall", "f1"):
        frame[f"vlm_{metric}"] = base[metric]
        frame[f"mas_{metric}"] = mode[metric]
        frame[f"delta_{metric}"] = mode[metric] - base[metric]
    frame["true_share"] = vlm_primitives["actual_primitive"].value_counts(normalize=True).reindex(PRIMITIVES).fillna(0.0)
    frame["vlm_predicted_share"] = vlm_primitives["predicted_primitive"].value_counts(normalize=True).reindex(PRIMITIVES).fillna(0.0)
    frame["mas_predicted_share"] = primitives["predicted_primitive"].value_counts(normalize=True).reindex(PRIMITIVES).fillna(0.0)
    frame["delta_predicted_share"] = frame["mas_predicted_share"] - frame["vlm_predicted_share"]
    return frame.reset_index().rename(columns={"index": "label"})


def _subject_primitive_counts(primitives: pd.DataFrame) -> pd.DataFrame:
    counts = build_count_aggregations(count_video_primitives(primitives[COUNT_COLUMNS]))["subject_primitive"]
    return counts[["subject_id", "primitive", "actual_count", "predicted_count"]]


def count_changes(primitives: pd.DataFrame, vlm_primitives: pd.DataFrame) -> pd.DataFrame:
    """Per subject and primitive; the `total` rows sum the primitives of a subject."""
    base = _subject_primitive_counts(vlm_primitives).rename(columns={"predicted_count": "vlm_count"})
    mode = _subject_primitive_counts(primitives).rename(columns={"predicted_count": "mas_count"})
    frame = base.merge(mode.drop(columns="actual_count"), on=["subject_id", "primitive"], how="outer").fillna(0)
    totals = frame.groupby("subject_id", as_index=False)[["actual_count", "vlm_count", "mas_count"]].sum()
    frame = pd.concat([frame, totals.assign(primitive="total")], ignore_index=True)
    frame["vlm_abs_error"] = (frame["vlm_count"] - frame["actual_count"]).abs()
    frame["mas_abs_error"] = (frame["mas_count"] - frame["actual_count"]).abs()
    frame["abs_error_change"] = frame["mas_abs_error"] - frame["vlm_abs_error"]
    frame["mas_minus_vlm_count"] = frame["mas_count"] - frame["vlm_count"]
    frame["effect"] = np.select(
        [frame["abs_error_change"] < 0, frame["abs_error_change"] > 0], ["improved", "worse"], "unchanged"
    )
    return frame.sort_values(["primitive", "subject_id"]).reset_index(drop=True)


def _subject_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, group in frame.groupby("subject_id"):
        per_class = per_class_summary(group)
        released = group[group["released"]]
        unsafe = released["predicted_label_id"].isin(ACTIVE_IDS) & (released["actual_label_id"] == REST_ID)
        rows.append(
            {
                "subject_id": subject_id,
                "frames": len(group),
                "accuracy": float((group["predicted_label_id"] == group["actual_label_id"]).mean()),
                "macro_f1": float(per_class.loc[per_class["support"] > 0, "f1"].mean()),
                "unsafe_rate": float(unsafe.mean()) if len(released) else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("subject_id")


def subject_deltas(primitives: pd.DataFrame, vlm_primitives: pd.DataFrame) -> pd.DataFrame:
    base, mode = _subject_scores(vlm_primitives), _subject_scores(primitives)
    frame = pd.DataFrame({"frames": base["frames"]})
    for metric in ("accuracy", "macro_f1", "unsafe_rate"):
        frame[f"vlm_{metric}"] = base[metric]
        frame[f"mas_{metric}"] = mode[metric]
        frame[f"delta_{metric}"] = mode[metric] - base[metric]
    return frame.reset_index()


def headline(summary: dict, vlm_summary: dict) -> pd.DataFrame:
    rows = [
        {
            "metric": metric,
            "vlm": vlm_summary.get(metric, np.nan),
            "mas": summary.get(metric, np.nan),
            "delta": summary.get(metric, np.nan) - vlm_summary.get(metric, np.nan),
        }
        for metric in HEADLINE_METRICS
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ figures


def _bar_panel(axis, labels, values, title, fmt="{:+.0f}") -> None:
    positions = np.arange(len(labels))
    colors = [plots.STATUS["approve"] if v > 0 else plots.STATUS["block"] if v < 0 else plots.NEUTRAL for v in values]
    axis.barh(positions, values, height=0.6, color=colors, edgecolor=plots.SURFACE, linewidth=2)
    axis.axvline(0, color=plots.INK_SECONDARY, linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    axis.set_title(title, loc="left")
    span = max(float(np.nanmax(np.abs(values))) if len(values) else 1.0, 1e-9)
    for position, value in zip(positions, values):
        axis.text(value + (0.02 if value >= 0 else -0.02) * span, position, fmt.format(value),
                  va="center", ha="left" if value >= 0 else "right", fontsize=8, color=plots.INK)
    axis.set_xlim(-span * 1.3, span * 1.3)


def plot_change_attribution(attribution: pd.DataFrame, path: Path, title: str) -> Path:
    plots._style()
    figure, axis = plots.plt.subplots(figsize=(8, 4.2))
    _bar_panel(axis, attribution["stage"].tolist(), attribution["net_correct_frames"].to_numpy(float),
               "Net frames made correct (after − before), per stage")
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    return plots._save(figure, path)


def plot_class_f1_delta(classes: pd.DataFrame, path: Path, title: str) -> Path:
    plots._style()
    figure, axes = plots.plt.subplots(1, 2, figsize=(10, 3.4))
    _bar_panel(axes[0], classes["label"].tolist(), classes["delta_f1"].to_numpy(float), "F1 change vs VLM", "{:+.4f}")
    _bar_panel(axes[1], classes["label"].tolist(), classes["delta_predicted_share"].to_numpy(float),
               "Change in share of frames predicted as the class", "{:+.4f}")
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    return plots._save(figure, path)


def plot_count_error(counts: pd.DataFrame, path: Path, title: str) -> Path:
    plots._style()
    totals = counts[counts["primitive"] == "total"].sort_values("subject_id")
    figure, axis = plots.plt.subplots(figsize=(max(6, 0.6 * len(totals) + 2), 4))
    positions = np.arange(len(totals))
    axis.bar(positions - 0.2, totals["vlm_abs_error"], width=0.4, color=plots.mode_color("vlm"), label="VLM")
    axis.bar(positions + 0.2, totals["mas_abs_error"], width=0.4, color=plots.mode_color("mas"), label="MAS")
    axis.set_xticks(positions, totals["subject_id"], rotation=45)
    axis.set_ylabel("|predicted − actual| primitive count")
    axis.set_title("Total primitive-count error per subject (lower is better)", loc="left")
    axis.legend()
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    return plots._save(figure, path)


def plot_subject_deltas(deltas: pd.DataFrame, path: Path, title: str) -> Path:
    plots._style()
    ordered = deltas.sort_values("subject_id")
    figure, axes = plots.plt.subplots(1, 2, figsize=(11, max(3, 0.35 * len(ordered) + 1.5)))
    _bar_panel(axes[0], ordered["subject_id"].tolist(), ordered["delta_accuracy"].to_numpy(float),
               "Accuracy change vs VLM", "{:+.4f}")
    _bar_panel(axes[1], ordered["subject_id"].tolist(), -ordered["delta_unsafe_rate"].to_numpy(float),
               "Unsafe-rate reduction vs VLM (positive is safer)", "{:+.4f}")
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    return plots._save(figure, path)


def plot_safety_tradeoff(deltas: pd.DataFrame, path: Path, title: str) -> Path:
    plots._style()
    figure, axis = plots.plt.subplots(figsize=(6.5, 4.5))
    axis.scatter(-deltas["delta_unsafe_rate"], deltas["delta_accuracy"], s=40,
                 color=plots.mode_color("mas"), edgecolor=plots.SURFACE, linewidth=1.5, zorder=3)
    for row in deltas.itertuples(index=False):
        axis.annotate(row.subject_id, (-row.delta_unsafe_rate, row.delta_accuracy), fontsize=7,
                      color=plots.INK_SECONDARY, xytext=(3, 3), textcoords="offset points")
    axis.axhline(0, color=plots.INK_SECONDARY, linewidth=0.8)
    axis.axvline(0, color=plots.INK_SECONDARY, linewidth=0.8)
    axis.set_xlabel("Unsafe-rate reduction vs VLM (right is safer)")
    axis.set_ylabel("Accuracy change vs VLM (up is better)")
    axis.set_title("Per subject: safety gained vs accuracy paid", loc="left")
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    return plots._save(figure, path)


def analyse_mode(
    *,
    mode: str,
    records: pd.DataFrame,
    primitives: pd.DataFrame,
    vlm_primitives: pd.DataFrame,
    summary: dict,
    vlm_summary: dict,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Write the tables and figures above for one MAS mode against the VLM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = stage_frame(records, primitives)
    tables = {
        "change_attribution": change_attribution(stages),
        "revision_outcomes": revision_outcomes(stages),
        "class_changes": class_changes(primitives, vlm_primitives),
        "count_changes_by_subject": count_changes(primitives, vlm_primitives),
        "subject_deltas": subject_deltas(primitives, vlm_primitives),
        "headline": headline(summary, vlm_summary),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    label = plots.MODE_LABELS.get(mode, mode)
    plot_change_attribution(tables["change_attribution"], output_dir / "change_attribution.png", f"{label}: where the agents changed the VLM's cue")
    plot_class_f1_delta(tables["class_changes"], output_dir / "class_f1_delta.png", f"{label} vs VLM, per class")
    plot_count_error(tables["count_changes_by_subject"], output_dir / "count_error_vs_vlm.png", f"{label} vs VLM, primitive counts")
    plot_subject_deltas(tables["subject_deltas"], output_dir / "subject_accuracy_delta.png", f"{label} vs VLM, per subject")
    plot_safety_tradeoff(tables["subject_deltas"], output_dir / "safety_tradeoff.png", f"{label} vs VLM")
    return tables
