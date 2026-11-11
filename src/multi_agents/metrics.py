"""Measures for the VLM baseline and every MAS mode, computed identically.

Ground truth is joined here, after a run, from the window log; agents never see it.
Every frame-level measure scores all five horizons of every window ("frames"); a frame
is "released" when its cue is delivered, and a withheld frame keeps its planned cue.

  unsafe rate            released active cue (reach/reposition/transport) on a frame
                         whose ground truth is rest, / released frames
  coverage               released frames / frames
  withheld correct+safe  withheld frames whose cue was correct and not unsafe, / withheld
  count error            primitive segments (run-length collapsed) per subject, summed
                         over clips, predicted against actual
  edit score, AER        per clip, on run-length-collapsed label sequences, then averaged
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

if TYPE_CHECKING:
    from multi_agents.data import ExerciseSource
    from multi_agents.priors import PopulationPriors
from multi_agents.settings import ACTIVE_IDS, HORIZONS, NUM_PRIMITIVES, PRIMITIVES, REST_ID, MASConfig
from config import MASK_LABEL_ID

# Modes whose UE-FMA is the evaluation agent's pooled patient estimate (patient memory on).
MEMORY_MODES = ("mas", "mas_no_planning", "mas_no_safety")
# The agent stages scored in change_attribution.csv: (name, cue before, cue after).
STAGES = (
    ("planning", "model_label_id", "plan_label_id"),
    ("safety", "plan_label_id", "predicted_label_id"),
)


# ------------------------------------------------------------------ frames


def primitive_frame(records: pd.DataFrame, source: ExerciseSource) -> pd.DataFrame:
    """One row per scored (window, horizon) with ground truth."""
    label_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for record in records.itertuples(index=False):
        key = f"{record.subject_id}/{record.clip_id}"
        if key not in label_cache:
            label_cache[key] = source.labels(source.find(record.clip_id, record.subject_id))
        actual_ids = label_cache[key][record.window_index]
        for horizon in range(HORIZONS):
            actual = int(actual_ids[horizon])
            if actual == MASK_LABEL_ID:
                continue
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "clip_id": record.clip_id,
                    "horizon": horizon,
                    "absolute_frame_index": int(record.target_frame_index) + horizon,
                    "actual_label_id": actual,
                    "predicted_label_id": PRIMITIVES.index(record.final_action[horizon]),
                    "model_label_id": PRIMITIVES.index(record.model_action[horizon]),
                    "plan_label_id": PRIMITIVES.index(record.plan_action[horizon]),
                    "released": bool(record.released),
                }
            )
    return pd.DataFrame(rows)


def exercise_frame(
    records: pd.DataFrame,
    source: ExerciseSource,
    exercise_records: dict[str, list[dict]] | None,
    config: MASConfig,
) -> pd.DataFrame:
    """Per-exercise calibrated UE-FMA: the evaluation agent's running estimate in MAS
    modes, the exercise's mean window score otherwise."""
    rows = []
    for (subject_id, clip_id), windows in records.groupby(["subject_id", "clip_id"], sort=False):
        scores = windows["model_ue_fma"].to_numpy(dtype=float)
        scores = scores[np.isfinite(scores)]
        rows.append(
            {
                "subject_id": subject_id,
                "clip_id": clip_id,
                "exercise_index": int(windows["exercise_index"].iloc[0]),
                "ue_model_exercise_raw": float(scores.mean()) if scores.size else np.nan,
                "ue_true": source.true_uefma(subject_id),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["subject_id", "exercise_index"]).reset_index(drop=True)
    frame["ue_model_exercise"] = config.calibrate_uefma(frame["ue_model_exercise_raw"].to_numpy(dtype=float))
    frame["ue_model_running_mean"] = frame.groupby("subject_id")["ue_model_exercise"].transform(
        lambda scores: scores.expanding().mean()
    )
    if exercise_records:
        stored = {
            (subject, record["clip_id"]): record["ue_fma_estimate_after"]
            for subject, records_ in exercise_records.items()
            for record in records_
        }
        frame["ue_estimate"] = [stored.get(key, np.nan) for key in zip(frame["subject_id"], frame["clip_id"])]
    else:
        frame["ue_estimate"] = frame["ue_model_exercise"]
    frame["ue_abs_error_estimate"] = (frame["ue_estimate"] - frame["ue_true"]).abs()
    frame["ue_abs_error_running_mean"] = (frame["ue_model_running_mean"] - frame["ue_true"]).abs()
    return frame


def uefma_subject_frame(
    records: pd.DataFrame, exercises: pd.DataFrame, source: ExerciseSource, mode: str, config: MASConfig
) -> pd.DataFrame:
    """Subject-level calibrated UE-FMA: the final patient estimate in memory modes, the
    mean over all of the subject's windows otherwise."""
    rows = []
    for subject_id, windows in records.groupby("subject_id"):
        if mode in MEMORY_MODES:
            last = exercises[exercises["subject_id"] == subject_id].sort_values("exercise_index").iloc[-1]
            predicted = float(last["ue_estimate"])
        else:
            predicted = config.calibrate_uefma(float(np.nanmean(windows["model_ue_fma"].to_numpy(dtype=float))))
        rows.append({"subject_id": subject_id, "actual_ue_fma": source.true_uefma(subject_id), "predicted_ue_fma": predicted})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ measures


def _safe_pearson(actual, predicted) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(p).size < 2:
        return float("nan")
    return float(pearsonr(a, p).statistic)


def classification(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual_label_id"].to_numpy(dtype=int)
    predicted = frame["predicted_label_id"].to_numpy(dtype=int)
    labels = list(range(NUM_PRIMITIVES))
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
    }


def subject_primitive_counts(primitives: pd.DataFrame) -> pd.DataFrame:
    """Actual and predicted segment counts per subject and primitive, plus a `total` row.

    A segment is a run of consecutive frames of one clip with the same label.
    """
    frame = primitives.sort_values(["subject_id", "clip_id", "absolute_frame_index"])
    new_run = (
        (frame["subject_id"] != frame["subject_id"].shift())
        | (frame["clip_id"] != frame["clip_id"].shift())
        | (frame["absolute_frame_index"].diff() != 1)
    )
    subjects = sorted(frame["subject_id"].unique())
    index = pd.MultiIndex.from_product([subjects, range(NUM_PRIMITIVES)], names=["subject_id", "label_id"])
    counts = {}
    for kind in ("actual", "predicted"):
        labels = frame[f"{kind}_label_id"]
        starts = frame.loc[new_run | (labels != labels.shift()), ["subject_id", f"{kind}_label_id"]]
        counts[f"{kind}_count"] = (
            starts.groupby(["subject_id", f"{kind}_label_id"]).size().rename_axis(index.names).reindex(index, fill_value=0)
        )
    table = pd.DataFrame(counts).reset_index()
    table["primitive"] = [PRIMITIVES[label] for label in table["label_id"]]
    totals = table.groupby("subject_id", as_index=False)[["actual_count", "predicted_count"]].sum()
    table = pd.concat([table.drop(columns="label_id"), totals.assign(primitive="total")], ignore_index=True)
    table["abs_error"] = (table["predicted_count"] - table["actual_count"]).abs()
    return table[["subject_id", "primitive", "actual_count", "predicted_count", "abs_error"]]


def _levenshtein(actual: np.ndarray, predicted: np.ndarray) -> int:
    """Row-by-row DP; the insertion chain is a running minimum, so each row is vectorized."""
    n = len(predicted)
    index = np.arange(n + 1)
    previous = index.copy()
    for i, value in enumerate(actual, start=1):
        candidate = np.empty(n + 1, dtype=np.int64)
        candidate[0] = i
        candidate[1:] = np.minimum(previous[1:] + 1, previous[:-1] + (predicted != value))
        previous = np.minimum.accumulate(candidate - index) + index
    return int(previous[-1])


def _collapse(values: np.ndarray) -> np.ndarray:
    return values[np.r_[True, values[1:] != values[:-1]]] if len(values) else values


def sequence_scores(primitives: pd.DataFrame) -> dict[str, float]:
    """Clip means of the edit score and the action error rate (AER)."""
    edit, aer = [], []
    frame = primitives.sort_values(["subject_id", "clip_id", "absolute_frame_index"])
    for _, clip in frame.groupby(["subject_id", "clip_id"], sort=False):
        actual = _collapse(clip["actual_label_id"].to_numpy())
        predicted = _collapse(clip["predicted_label_id"].to_numpy())
        if len(actual) == 0:
            continue
        distance = _levenshtein(actual, predicted)
        edit.append(1.0 - distance / max(len(actual), len(predicted)))
        aer.append(distance / len(actual))
    return {"edit_score": float(np.mean(edit)), "action_error_rate": float(np.mean(aer))}


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


def summary_metrics(
    *,
    mode: str,
    primitives: pd.DataFrame,
    records: pd.DataFrame,
    exercises: pd.DataFrame,
    uefma_subjects: pd.DataFrame,
    counts: pd.DataFrame,
    config: MASConfig,
    priors: PopulationPriors,
) -> dict[str, Any]:
    """One row of metrics.csv."""
    row: dict[str, Any] = {"mode": mode}
    for suffix, selected in (
        ("", primitives),
        ("_h0", primitives[primitives["horizon"] == 0]),
        ("_h4", primitives[primitives["horizon"] == HORIZONS - 1]),
    ):
        row.update({f"{key}{suffix}": value for key, value in classification(selected).items()})
    row.update(sequence_scores(primitives))

    totals = counts[counts["primitive"] == "total"]
    valid = totals["actual_count"] != 0
    row["count_mae"] = float(totals["abs_error"].mean())
    row["count_mape"] = float((100.0 * totals.loc[valid, "abs_error"] / totals.loc[valid, "actual_count"]).mean())
    row["count_pearson_r"] = _safe_pearson(totals["actual_count"], totals["predicted_count"])

    error = uefma_subjects["predicted_ue_fma"] - uefma_subjects["actual_ue_fma"]
    row["ue_fma_exercise_mae"] = float(exercises["ue_abs_error_estimate"].mean())
    row["ue_fma_running_mean_mae"] = float(exercises["ue_abs_error_running_mean"].mean())
    row["ue_fma_mae"] = float(error.abs().mean())
    row["ue_fma_rmse"] = float(np.sqrt((error**2).mean()))
    row["ue_fma_pearson_r"] = _safe_pearson(uefma_subjects["actual_ue_fma"], uefma_subjects["predicted_ue_fma"])

    released = primitives[primitives["released"]]
    withheld = primitives[~primitives["released"]]
    unsafe = primitives["released"] & primitives["predicted_label_id"].isin(ACTIVE_IDS) & (primitives["actual_label_id"] == REST_ID)
    withheld_correct_safe = (withheld["predicted_label_id"] == withheld["actual_label_id"]) & ~(
        withheld["predicted_label_id"].isin(ACTIVE_IDS) & (withheld["actual_label_id"] == REST_ID)
    )
    failures = records["coordination_failure"].dropna().astype(str)
    row["unsafe_rate"] = float(unsafe.sum() / max(len(released), 1))
    row["coverage"] = float(primitives["released"].mean())
    row["implausible_transition_rate"] = _implausible_rate(primitives, priors, config.implausible_transition)
    row["withheld_frames"] = len(withheld)
    row["withheld_correct_and_safe_rate"] = float(withheld_correct_safe.mean()) if len(withheld) else 0.0
    row["released_violating_c1"] = int(failures.str.startswith("released_cue_not_permitted").sum())
    return row


def change_attribution(primitives: pd.DataFrame) -> list[dict[str, Any]]:
    """Frames each agent stage changed, and the accuracy on them before and after."""
    rows = []
    truth = primitives["actual_label_id"]
    for stage, before, after in STAGES:
        mask = primitives[before] != primitives[after]
        changed = int(mask.sum())
        rows.append(
            {
                "stage": stage,
                "frames_changed": changed,
                "share_of_frames": float(mask.mean()),
                "accuracy_before": float((primitives.loc[mask, before] == truth[mask]).mean()) if changed else np.nan,
                "accuracy_after": float((primitives.loc[mask, after] == truth[mask]).mean()) if changed else np.nan,
            }
        )
    return rows


def reference_agreement(replay: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Join scored frames once each; retain both input sizes to expose missing matches."""
    keys = ["subject_id", "clip_id", "absolute_frame_index"]
    reference = reference.copy()
    if "absolute_frame_index" not in reference:
        reference["absolute_frame_index"] = reference["target_frame_index"] + reference["horizon"]
    if "predicted_label_id" not in reference:
        reference["predicted_label_id"] = reference["predicted_primitive"].map(PRIMITIVES.index)
    joined = replay[keys + ["predicted_label_id"]].merge(
        reference[keys + ["predicted_label_id"]], on=keys, suffixes=("", "_reference"),
        validate="one_to_one",
    )
    return {
        "status": "available" if len(joined) else "no_matching_frames",
        "replay_frames": len(replay), "reference_frames": len(reference),
        "matched_frames": len(joined),
        "agreement": float((joined["predicted_label_id"] == joined["predicted_label_id_reference"]).mean()),
    }
