"""Segment-sequence metrics: edit score, action error rate, relative counting error.

Computed per clip on de-duplicated label sequences (consecutive repeats collapsed)
and averaged over clips:

    d   = Levenshtein distance(actual, predicted)
    ES  = 1 - d / max(len(actual), len(predicted))                    higher is better
    AER = d / len(actual)                                             lower is better
    RCE = sum_c |count_actual(c) - count_predicted(c)| / len(actual)  lower is better

The ground-truth sequence is every scored frame's label. Two predicted sequences:
  frame   every frame's prediction (the five positions tile the timeline; the same
          timeline as the count metrics)
  window  one prediction per window, the current frame (h = 0), every 5th frame

`evaluate.py` computes these for every mode (`sequence_metrics_clips.csv`, and the clip
means in comparison.csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multi_agents.settings import NUM_PRIMITIVES

LEVELS = ("frame", "window")
METRICS = ("edit_score", "action_error_rate", "relative_counting_error")
COLUMNS = ["subject_id", "clip_id", "horizon", "absolute_frame_index", "actual_label_id", "predicted_label_id"]


def deduplicate(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    keep = np.r_[True, values[1:] != values[:-1]]
    return values[keep]


def levenshtein(actual: np.ndarray, predicted: np.ndarray) -> int:
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


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    distance = levenshtein(actual, predicted)
    counts = np.abs(np.bincount(actual, minlength=NUM_PRIMITIVES) - np.bincount(predicted, minlength=NUM_PRIMITIVES)).sum()
    return {
        "segments_actual": len(actual),
        "segments_predicted": len(predicted),
        "edit_score": 1.0 - distance / max(len(actual), len(predicted)),
        "action_error_rate": distance / len(actual),
        "relative_counting_error": counts / len(actual),
    }


def clip_sequence_frame(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per clip and level from scored frames (the `COLUMNS` of a primitives frame
    or of a `<split>_combined_window_predictions.csv`)."""
    frames = frames[COLUMNS].sort_values(["subject_id", "clip_id", "absolute_frame_index"])
    rows = []
    for (subject_id, clip_id), clip in frames.groupby(["subject_id", "clip_id"], sort=False):
        actual = deduplicate(clip["actual_label_id"].to_numpy())
        if len(actual) == 0:
            continue
        for level, predicted in (
            ("frame", clip["predicted_label_id"].to_numpy()),
            ("window", clip.loc[clip["horizon"] == 0, "predicted_label_id"].to_numpy()),
        ):
            rows.append({"subject_id": subject_id, "clip_id": clip_id, "level": level, **scores(actual, deduplicate(predicted))})
    return pd.DataFrame(rows)


def summary(clips: pd.DataFrame) -> dict[str, float]:
    """Clip means per level: `edit_score_frame`, ..., `relative_counting_error_window`."""
    result = {}
    for level in LEVELS:
        selected = clips[clips["level"] == level] if len(clips) else clips
        for metric in METRICS:
            result[f"{metric}_{level}"] = float(selected[metric].mean()) if len(selected) else np.nan
    return result
