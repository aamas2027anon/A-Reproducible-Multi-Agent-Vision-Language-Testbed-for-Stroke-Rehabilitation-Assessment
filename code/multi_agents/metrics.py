"""Measures for the VLM baseline and every MAS mode, computed identically.

Ground truth is joined here, after a run, from the window log. Primitive and counting
metrics reuse the training pipeline's own functions on training-schema frames, so
the numbers are directly comparable with (and plottable next to) the training runs.

Definitions (horizon-level rows = one frame each; "released" = cue delivered):
  unsafe recommendation rate  released active cue (reach/reposition/transport) on a
                              frame whose ground truth is rest, / released frames
  hazard cue rate (GT-free)   released active cue where the model's P(rest) >= HAZARD_P_REST
  unnecessary escalation      withheld frames whose withheld cue was correct and not
                              hazardous, / withheld frames
  coordination failures       released cues that break a safety rule, disagreements
                              resolved to the wrong agent
  latency                     wall-clock ms per agent per window and per exercise
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from multi_agents import sequence_metrics
from multi_agents.data import ExerciseSource
from multi_agents.priors import PopulationPriors
from multi_agents.settings import ACTIVE_IDS, HORIZONS, PRIMITIVES, REST_ID, MASConfig
from multi_agents.train_bridge import (
    bootstrap_mean_ci,
    build_count_aggregations,
    classification_summary,
    count_video_primitives,
    monte_carlo_sign_flip_pvalue,
    train_config,
)

PRIMITIVE_EXPORT_COLUMNS = (
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
)
UEFMA_EXPORT_COLUMNS = ("subject_id", "actual_ue_fma", "predicted_ue_fma", "absolute_error", "run_name", "fold_id", "split")
# Modes whose UE-FMA is the evaluation agent's pooled patient estimate (patient memory on).
MEMORY_MODES = ("mas", "mas_no_planning", "mas_no_safety")
# P(rest) above which a released movement cue counts toward the hazard cue rate.
HAZARD_P_REST = 0.35


def primitive_frame(records: pd.DataFrame, source: ExerciseSource, mode: str, split: str) -> pd.DataFrame:
    """One row per (window, horizon) with ground truth, in the training export schema."""
    label_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for record in records.itertuples(index=False):
        key = f"{record.subject_id}/{record.clip_id}"
        if key not in label_cache:
            label_cache[key] = source.labels(source.find(record.clip_id, record.subject_id))
        actual_ids = label_cache[key][record.window_index]
        for horizon in range(HORIZONS):
            actual = int(actual_ids[horizon])
            if actual == train_config.MASK_LABEL_ID:
                continue
            predicted = record.final_action[horizon]
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "clip_id": record.clip_id,
                    "task_name": record.task_name,
                    "target_frame_index": int(record.target_frame_index),
                    "horizon": horizon,
                    "actual_primitive": PRIMITIVES[actual],
                    "predicted_primitive": predicted,
                    "prediction_probability": float(record.final_confidence[horizon]),
                    "run_name": mode,
                    "fold_id": -1,
                    "split": split,
                    "actual_label_id": actual,
                    "predicted_label_id": PRIMITIVES.index(predicted),
                    "absolute_frame_index": int(record.target_frame_index) + horizon,
                    "prediction_position": train_config.POSITION_NAMES[horizon],
                    "exercise_index": int(record.exercise_index),
                    "window_index": int(record.window_index),
                    "released": bool(record.released),
                    "status": record.status,
                    "model_label_id": PRIMITIVES.index(record.model_action[horizon]),
                    "plan_label_id": PRIMITIVES.index(record.plan_action[horizon]),
                    "model_p_rest": float(record.model_p_rest[horizon]),
                    "disagreement": bool(record.disagreement),
                }
            )
    return pd.DataFrame(rows)


def exercise_frame(
    records: pd.DataFrame,
    primitives: pd.DataFrame,
    source: ExerciseSource,
    mode: str,
    exercise_records: dict[str, list[dict]] | None,
    config: MASConfig,
) -> pd.DataFrame:
    """Per-exercise trajectory rows: UE-FMA estimates, accuracy, decisions.

    Every UE-FMA column is calibrated (`MASConfig.calibrate_uefma`): `ue_model_exercise`
    is the calibrated mean of the exercise's window scores (raw mean in
    `ue_model_exercise_raw`), and `ue_estimate` is the evaluation agent's mu_cal for
    memory modes or the calibrated exercise mean otherwise. `ue_estimate_raw` is the same
    estimate uncalibrated, for comparison with raw-scale numbers (training, older runs).
    """
    rows = []
    for (subject_id, clip_id), windows in records.groupby(["subject_id", "clip_id"], sort=False):
        frames = primitives[(primitives["subject_id"] == subject_id) & (primitives["clip_id"] == clip_id)]
        scores = windows["model_ue_fma"].to_numpy(dtype=float)
        scores = scores[np.isfinite(scores)]
        row = {
            "mode": mode,
            "subject_id": subject_id,
            "clip_id": clip_id,
            "task_name": windows["task_name"].iloc[0],
            "exercise_index": int(windows["exercise_index"].iloc[0]),
            "n_windows": len(windows),
            "accuracy": float((frames["actual_label_id"] == frames["predicted_label_id"]).mean()),
            "model_accuracy": float((frames["actual_label_id"] == frames["model_label_id"]).mean()),
            "coverage": float(windows["released"].mean()),
            "approve": int((windows["status"] == "approve").sum()),
            "revise": int((windows["status"] == "revise").sum()),
            "escalate": int((windows["status"] == "escalate").sum()),
            "block": int((windows["status"] == "block").sum()),
            "ue_model_exercise_raw": float(scores.mean()) if scores.size else np.nan,
            "ue_true": source.true_uefma(subject_id),
            "ue_estimate": np.nan,
            "ue_estimate_raw": np.nan,
            "ue_ci95": np.nan,
            "clinician_flag": False,
        }
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["subject_id", "exercise_index"]).reset_index(drop=True)
    frame["ue_model_exercise"] = config.calibrate_uefma(frame["ue_model_exercise_raw"].to_numpy(dtype=float))
    frame["ue_model_running_mean"] = frame.groupby("subject_id")["ue_model_exercise"].transform(
        lambda s: s.expanding().mean()
    )
    if exercise_records:
        stored = {
            (subject, record["clip_id"]): record
            for subject, records_ in exercise_records.items()
            for record in records_
        }
        for index, row in frame.iterrows():
            record = stored.get((row["subject_id"], row["clip_id"]))
            if record is not None:
                frame.loc[index, "ue_estimate"] = record["ue_fma_estimate_after"]
                raw = record.get("ue_fma_estimate_after_raw")
                frame.loc[index, "ue_estimate_raw"] = np.nan if raw is None else raw
                frame.loc[index, "ue_ci95"] = record["ue_fma_ci95"]
                frame.loc[index, "clinician_flag"] = record["clinician_flag"]
    else:
        frame["ue_estimate"] = frame["ue_model_exercise"]
        frame["ue_estimate_raw"] = frame["ue_model_exercise_raw"]
    frame["ue_abs_error_estimate"] = (frame["ue_estimate"] - frame["ue_true"]).abs()
    frame["ue_abs_error_estimate_raw"] = (frame["ue_estimate_raw"] - frame["ue_true"]).abs()
    frame["ue_abs_error_model_exercise"] = (frame["ue_model_exercise"] - frame["ue_true"]).abs()
    frame["ue_abs_error_model_running_mean"] = (frame["ue_model_running_mean"] - frame["ue_true"]).abs()
    return frame


def uefma_subject_frame(
    records: pd.DataFrame, exercises: pd.DataFrame, source: ExerciseSource, mode: str, split: str, config: MASConfig
) -> pd.DataFrame:
    """Subject-level UE-FMA in the training schema, calibrated in every mode.

    Memoryless modes use the training convention (mean over all windows), calibrated;
    memory modes report the evaluation agent's final patient estimate mu_cal. The extra
    column `predicted_ue_fma_raw` is the uncalibrated prediction; it is not part of the
    exported training-schema CSV (`UEFMA_EXPORT_COLUMNS`).
    """
    rows = []
    for subject_id, windows in records.groupby("subject_id"):
        if mode in MEMORY_MODES:
            last = exercises[exercises["subject_id"] == subject_id].sort_values("exercise_index").iloc[-1]
            predicted = float(last["ue_estimate"])
            predicted_raw = float(last["ue_estimate_raw"])
        else:
            scores = windows["model_ue_fma"].to_numpy(dtype=float)
            predicted_raw = float(np.nanmean(scores))
            predicted = config.calibrate_uefma(predicted_raw)
        actual = source.true_uefma(subject_id)
        rows.append(
            {
                "subject_id": subject_id,
                "actual_ue_fma": actual,
                "predicted_ue_fma": predicted,
                "absolute_error": abs(actual - predicted),
                "run_name": mode,
                "fold_id": -1,
                "split": split,
                "predicted_ue_fma_raw": predicted_raw,
            }
        )
    return pd.DataFrame(rows)[[*UEFMA_EXPORT_COLUMNS, "predicted_ue_fma_raw"]]


def combined_frame(
    primitives: pd.DataFrame, records: pd.DataFrame, exercises: pd.DataFrame, mode: str, config: MASConfig
) -> pd.DataFrame:
    """Primitive rows plus calibrated window UE-FMA (memoryless modes) or the patient estimate mu_cal (memory modes)."""
    windows = records[["subject_id", "clip_id", "window_index", "model_ue_fma"]]
    combined = primitives.merge(
        windows, on=["subject_id", "clip_id", "window_index"], how="left", validate="many_to_one"
    )
    combined["actual_ue_fma"] = combined["subject_id"].map(
        exercises.groupby("subject_id")["ue_true"].first()
    )
    if mode in MEMORY_MODES:
        estimates = exercises.set_index(["subject_id", "clip_id"])["ue_estimate"]
        combined["predicted_ue_fma"] = estimates.reindex(
            pd.MultiIndex.from_frame(combined[["subject_id", "clip_id"]])
        ).to_numpy()
    else:
        combined["predicted_ue_fma"] = config.calibrate_uefma(combined["model_ue_fma"].to_numpy(dtype=float))
    return combined.drop(columns=["model_ue_fma"])


def _implausible_rate(primitives: pd.DataFrame, priors: PopulationPriors, threshold: float) -> float:
    released = primitives[primitives["released"]].sort_values(["subject_id", "clip_id", "absolute_frame_index"])
    pairs = implausible = 0
    for _, clip in released.groupby(["subject_id", "clip_id"], sort=False):
        frames = clip["absolute_frame_index"].to_numpy()
        labels = clip["predicted_label_id"].to_numpy()
        consecutive = np.diff(frames) == 1
        probabilities = priors.transition[labels[:-1], labels[1:]][consecutive]
        pairs += int(consecutive.sum())
        implausible += int((probabilities < threshold).sum())
    return implausible / pairs if pairs else np.nan


def _count_metrics(primitives: pd.DataFrame) -> dict[str, float]:
    columns = [*PRIMITIVE_EXPORT_COLUMNS, "actual_label_id", "predicted_label_id", "absolute_frame_index"]
    totals = build_count_aggregations(count_video_primitives(primitives[columns]))["subject_total"]
    error = totals["predicted_count"].to_numpy(float) - totals["actual_count"].to_numpy(float)
    actual, predicted = totals["actual_count"].to_numpy(float), totals["predicted_count"].to_numpy(float)
    correlation = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 and actual.std() and predicted.std() else np.nan
    return {"count_mae": float(np.mean(np.abs(error))), "count_pearson_r": correlation}


def summary_metrics(
    *,
    mode: str,
    primitives: pd.DataFrame,
    records: pd.DataFrame,
    exercises: pd.DataFrame,
    uefma_subjects: pd.DataFrame,
    exercise_records: dict[str, list[dict]] | None,
    inference_ms: dict[str, float],
    config: MASConfig,
    priors: PopulationPriors,
    sequence_clips: pd.DataFrame,
) -> dict[str, Any]:
    everything = classification_summary(primitives)
    current = classification_summary(primitives[primitives["horizon"] == 0])
    future = classification_summary(primitives[primitives["horizon"] > 0])
    next_t4 = classification_summary(primitives[primitives["horizon"] == HORIZONS - 1])
    step_ms = pd.to_numeric(records.get("step_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    released = primitives[primitives["released"]]
    withheld = primitives[~primitives["released"]]
    active = primitives["predicted_label_id"].isin(ACTIVE_IDS)
    gt_rest = primitives["actual_label_id"] == REST_ID
    unsafe = primitives["released"] & active & gt_rest
    hazard = primitives["released"] & active & (primitives["model_p_rest"] >= HAZARD_P_REST)
    withheld_correct_safe = (
        (withheld["predicted_label_id"] == withheld["actual_label_id"])
        & ~(withheld["predicted_label_id"].isin(ACTIVE_IDS) & (withheld["actual_label_id"] == REST_ID))
    )
    # Disagreement resolved to the wrong side: the rejected candidate was correct.
    disagreed = primitives[primitives["disagreement"] & (primitives["model_label_id"] != primitives["plan_label_id"])]
    wrong_side = int(
        (
            (disagreed["predicted_label_id"] != disagreed["actual_label_id"])
            & (
                (disagreed["model_label_id"] == disagreed["actual_label_id"])
                | (disagreed["plan_label_id"] == disagreed["actual_label_id"])
            )
        ).sum()
    )
    failures = records["coordination_failure"].dropna().value_counts().to_dict()

    latency: dict[str, float] = {}
    n_windows, n_exercises = len(records), len(exercises)
    if exercise_records:
        totals: dict[str, float] = {}
        for records_ in exercise_records.values():
            for record in records_:
                for agent, value in record["latency_ms"].items():
                    totals[agent] = totals.get(agent, 0.0) + value
        for agent, value in totals.items():
            latency[f"latency_ms_per_window_{agent}"] = value / n_windows
        agent_total = sum(v for k, v in totals.items() if k != "perception_model")
        latency["latency_ms_per_window_agents_total"] = agent_total / n_windows
        latency["latency_ms_per_exercise_total"] = sum(totals.values()) / n_exercises
    model_ms = sum(
        inference_ms.get(f"{subject}/{clip}", 0.0) for subject, clip in zip(exercises["subject_id"], exercises["clip_id"])
    )
    latency["latency_ms_per_window_perception_model"] = model_ms / n_windows
    latency.setdefault("latency_ms_per_exercise_total", model_ms / n_exercises)

    uefma_error = uefma_subjects["predicted_ue_fma"] - uefma_subjects["actual_ue_fma"]
    uefma_error_raw = uefma_subjects["predicted_ue_fma_raw"] - uefma_subjects["actual_ue_fma"]
    return {
        "mode": mode,
        "subjects": int(primitives["subject_id"].nunique()),
        "exercises": n_exercises,
        "windows": n_windows,
        "frames": len(primitives),
        "accuracy_future_primitive": next_t4["accuracy"],
        "macro_f1_future_primitive": next_t4["macro_f1"],
        "accuracy_all": everything["accuracy"],
        "macro_f1_all": everything["macro_f1"],
        "accuracy_current": current["accuracy"],
        "macro_f1_current": current["macro_f1"],
        "accuracy_future": future["accuracy"],
        "macro_f1_future": future["macro_f1"],
        "reach_f1_all": everything["reach_f1"],
        "reposition_f1_all": everything["reposition_f1"],
        **_count_metrics(primitives),
        # Edit score, action error rate, relative counting error (clip means; sequence_metrics.py).
        **sequence_metrics.summary(sequence_clips),
        "ue_fma_subject_mae": float(uefma_error.abs().mean()),
        "ue_fma_subject_rmse": float(np.sqrt((uefma_error**2).mean())),
        "ue_fma_subject_pearson_r": float(np.corrcoef(uefma_subjects["actual_ue_fma"], uefma_subjects["predicted_ue_fma"])[0, 1])
        if len(uefma_subjects) > 1
        else np.nan,
        "ue_fma_exercise_mae": float(exercises["ue_abs_error_estimate"].mean()),
        # Uncalibrated (raw-scale) counterparts, comparable with training and older runs.
        "ue_fma_subject_mae_raw": float(uefma_error_raw.abs().mean()),
        "ue_fma_exercise_mae_raw": float(exercises["ue_abs_error_estimate_raw"].mean()),
        "coverage_windows": float(records["released"].mean()),
        "coverage_frames": float(primitives["released"].mean()),
        "selective_accuracy": float((released["predicted_label_id"] == released["actual_label_id"]).mean())
        if len(released)
        else np.nan,
        "unsafe_recommendations": int(unsafe.sum()),
        "unsafe_recommendation_rate": float(unsafe.sum() / max(len(released), 1)),
        "hazard_cue_rate": float(hazard.sum() / max(len(released), 1)),
        "implausible_transition_rate": _implausible_rate(primitives, priors, config.implausible_transition),
        "withheld_frames": len(withheld),
        "unnecessary_escalations": int(withheld_correct_safe.sum()),
        "unnecessary_escalation_rate": float(withheld_correct_safe.mean()) if len(withheld) else 0.0,
        "unnecessary_escalations_per_1000_frames": 1000.0 * float(withheld_correct_safe.sum()) / len(primitives),
        "clinician_flags": int(exercises["clinician_flag"].sum()),
        "disagreement_windows": int(records["disagreement"].sum()),
        "coordination_failures_unpermitted_release": int(
            sum(v for k, v in failures.items() if str(k).startswith("released_cue_not_permitted"))
        ),
        "coordination_failures_wrong_side": wrong_side,
        "coordination_failure_rate": float(records["coordination_failure"].notna().sum() / max(n_windows, 1)),
        "step_ms_mean": float(step_ms.mean()) if len(step_ms) else np.nan,
        "step_ms_p95": float(step_ms.quantile(0.95)) if len(step_ms) else np.nan,
        **latency,
    }


def per_subject_metrics(primitives: pd.DataFrame, exercises: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, frame in primitives.groupby("subject_id"):
        summary = classification_summary(frame)
        released = frame[frame["released"]]
        active = released["predicted_label_id"].isin(ACTIVE_IDS)
        rows.append(
            {
                "subject_id": subject_id,
                "accuracy_all": summary["accuracy"],
                "macro_f1_all": summary["macro_f1"],
                "unsafe_recommendation_rate": float((active & (released["actual_label_id"] == REST_ID)).mean())
                if len(released)
                else 0.0,
                "ue_fma_exercise_mae": float(exercises.loc[exercises["subject_id"] == subject_id, "ue_abs_error_estimate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(per_subject: dict[str, pd.DataFrame], baseline: str = "vlm", seed: int = 9575) -> pd.DataFrame:
    """Per-subject paired deltas (mode - baseline) with bootstrap CI and sign-flip p."""
    if baseline not in per_subject:
        return pd.DataFrame()
    base = per_subject[baseline].set_index("subject_id")
    rows = []
    for mode, frame in per_subject.items():
        if mode == baseline:
            continue
        joined = frame.set_index("subject_id").join(base, rsuffix="_baseline", how="inner")
        for metric in ("accuracy_all", "macro_f1_all", "unsafe_recommendation_rate", "ue_fma_exercise_mae"):
            deltas = (joined[metric] - joined[f"{metric}_baseline"]).to_numpy(float)
            rng = np.random.default_rng(seed)
            low, high = bootstrap_mean_ci(deltas, rng) if len(deltas) > 1 else (np.nan, np.nan)
            p_value, _ = monte_carlo_sign_flip_pvalue(deltas, np.random.default_rng(seed)) if len(deltas) > 1 else (np.nan, 0)
            rows.append(
                {
                    "mode": mode,
                    "baseline": baseline,
                    "metric": metric,
                    "n_subjects": len(deltas),
                    "mean_delta": float(deltas.mean()) if len(deltas) else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "sign_flip_p": p_value,
                    "subjects_improved": int((deltas < 0).sum() if metric in ("unsafe_recommendation_rate", "ue_fma_exercise_mae") else (deltas > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def reference_agreement(vlm_primitives: pd.DataFrame, reference_csv: str) -> dict[str, Any]:
    """How closely the `vlm` mode reproduces the training pipeline's test predictions."""
    reference = pd.read_csv(reference_csv)
    keys = ["subject_id", "clip_id", "target_frame_index", "horizon"]
    merged = vlm_primitives.merge(reference[[*keys, "predicted_primitive"]], on=keys, suffixes=("", "_reference"))
    return {
        "reference": train_config.portable_path(reference_csv),
        "matched_rows": len(merged),
        "reference_rows": len(reference),
        "agreement": float((merged["predicted_primitive"] == merged["predicted_primitive_reference"]).mean())
        if len(merged)
        else np.nan,
    }
