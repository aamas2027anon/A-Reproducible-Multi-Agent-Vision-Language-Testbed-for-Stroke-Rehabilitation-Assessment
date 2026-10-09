"""History length: every mode scored on each patient's first N exercises.

The MAS is causal: exercises are replayed in a fixed session order and patient memory
only holds exercises already seen, so the first N exercises of a full run are decided
exactly as in a run limited to N exercises (`--limit-exercises N`). Slicing the full
run therefore gives the limited runs' metrics without running them.

For each mode and N (1 up to the smallest patient's exercise count, plus 2, 5, 10 and
"all"), the frame reports, over the patients' first N exercises:
  accuracy_all, macro_f1_all, accuracy_future_primitive   training's definitions, from
                                                          summed confusion matrices
  count_mae              per-subject |predicted - actual| primitive count (segments)
  unsafe_recommendation_rate   released movement cue on a true-rest frame, per released frame
  ue_fma_subject_mae     memory modes: the patient estimate mu_cal after exercise N;
                         other modes: the window-weighted mean of the first N exercises,
                         calibrated (MASConfig.calibrate_uefma)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multi_agents.mas_analysis import COUNT_COLUMNS
from multi_agents.metrics import MEMORY_MODES
from multi_agents.settings import ACTIVE_IDS, HORIZONS, NUM_PRIMITIVES, REST_ID, MASConfig
from multi_agents.train_bridge import count_video_primitives

MARKED_LIMITS = (2, 5, 10)
ALL = "all"


def _confusions(primitives: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Per (subject, exercise) 5x5 confusion counts, flattened into 25 columns."""
    frame = primitives if horizon is None else primitives[primitives["horizon"] == horizon]
    cell = (frame["actual_label_id"] * NUM_PRIMITIVES + frame["predicted_label_id"]).rename("cell")
    table = pd.crosstab([frame["subject_id"], frame["exercise_index"]], cell)
    return table.reindex(columns=range(NUM_PRIMITIVES * NUM_PRIMITIVES), fill_value=0)


def _scores(confusion: np.ndarray) -> tuple[float, float]:
    """Accuracy and macro-F1 (all five classes; F1 = 0 for an empty class) of one matrix."""
    matrix = confusion.reshape(NUM_PRIMITIVES, NUM_PRIMITIVES)
    total = matrix.sum()
    tp = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(NUM_PRIMITIVES), where=denominator > 0)
    return (float(tp.sum() / total) if total else np.nan), float(f1.mean())


def _limited(table: pd.DataFrame, limit: int) -> np.ndarray:
    return table[table.index.get_level_values("exercise_index") < limit].to_numpy().sum(axis=0)


def mode_rows(
    mode: str, primitives: pd.DataFrame, exercises: pd.DataFrame, limits: list[int | str], config: MASConfig
) -> list[dict]:
    all_frames = _confusions(primitives)
    future = _confusions(primitives, HORIZONS - 1)

    released = primitives[primitives["released"]]
    unsafe = released["predicted_label_id"].isin(ACTIVE_IDS) & (released["actual_label_id"] == REST_ID)
    safety = pd.DataFrame(
        {"released": 1, "unsafe": unsafe.astype(int), "subject_id": released["subject_id"], "exercise_index": released["exercise_index"]}
    ).groupby(["subject_id", "exercise_index"])[["released", "unsafe"]].sum()

    exercise_of = primitives.drop_duplicates(["subject_id", "clip_id"]).set_index(["subject_id", "clip_id"])["exercise_index"]
    videos = count_video_primitives(primitives[COUNT_COLUMNS])
    videos["exercise_index"] = exercise_of.reindex(pd.MultiIndex.from_frame(videos[["subject_id", "clip_id"]])).to_numpy()

    trajectory = exercises.sort_values(["subject_id", "exercise_index"])
    n_exercises = trajectory.groupby("subject_id")["exercise_index"].max() + 1

    rows = []
    for limit in limits:
        n = int(n_exercises.max()) if limit == ALL else int(limit)
        accuracy, macro_f1 = _scores(_limited(all_frames, n))
        future_accuracy, future_macro_f1 = _scores(_limited(future, n))
        counts = videos[videos["exercise_index"] < n].groupby("subject_id")[["actual_count", "predicted_count"]].sum()
        released_n = safety[safety.index.get_level_values("exercise_index") < n].sum()
        seen = trajectory[trajectory["exercise_index"] < n]
        if mode in MEMORY_MODES:
            estimate = seen.groupby("subject_id")["ue_estimate"].last()
        else:
            weights = seen["n_windows"] * seen["ue_model_exercise_raw"]
            raw = weights.groupby(seen["subject_id"]).sum() / seen.groupby("subject_id")["n_windows"].sum()
            estimate = pd.Series(config.calibrate_uefma(raw.to_numpy(dtype=float)), index=raw.index)
        truth = seen.groupby("subject_id")["ue_true"].first()
        rows.append(
            {
                "mode": mode,
                "limit": str(limit),
                "n_exercises": n,
                "marked": limit == ALL or limit in MARKED_LIMITS,
                "complete": bool((n_exercises >= n).all()) or limit == ALL,
                "exercises_scored": int(len(seen)),
                "accuracy_all": accuracy,
                "macro_f1_all": macro_f1,
                "accuracy_future_primitive": future_accuracy,
                "macro_f1_future_primitive": future_macro_f1,
                "count_mae": float((counts["predicted_count"] - counts["actual_count"]).abs().mean()),
                "unsafe_recommendation_rate": float(released_n["unsafe"] / released_n["released"]) if released_n["released"] else np.nan,
                "ue_fma_subject_mae": float((estimate - truth).abs().mean()),
            }
        )
    return rows


def exercise_limit_frame(
    primitives_by_mode: dict[str, pd.DataFrame], exercises_by_mode: dict[str, pd.DataFrame], config: MASConfig
) -> pd.DataFrame:
    """One row per mode and history length N (see the module docstring)."""
    any_exercises = next(iter(exercises_by_mode.values()))
    shortest = int((any_exercises.groupby("subject_id")["exercise_index"].max() + 1).min())
    limits: list[int | str] = sorted(set(range(1, shortest + 1)) | set(MARKED_LIMITS)) + [ALL]
    rows = []
    for mode, primitives in primitives_by_mode.items():
        rows += mode_rows(mode, primitives, exercises_by_mode[mode], limits, config)
    return pd.DataFrame(rows)
