"""Tune the planning agent, the safety cost and the UE-FMA calibration on the validation subjects (never on test).

    python -m multi_agents.tune --checkpoint <best.ckpt> --output tuned_config.json
    python -m multi_agents.evaluate --checkpoint <best.ckpt> --mas-config tuned_config.json

The model runs once per validation exercise (cached, like the evaluation). The grid is
then scored on CPU by replaying the planning filter and the patient-memory updates
exactly as the graph applies them, patient by patient in session order, without the
safety and coordination steps (they only act on top of the plan).

Selection: the highest next-primitive (frame t+4) accuracy, among settings whose
accuracy over all five frames is no worse than the model's by more than `TOLERANCE`.
The defaults (transition_strength = 0) reproduce the model, so a
setting is only chosen if it beats it.

Then, with those planning settings, the safety agent's cost c (`rest_cost`) is chosen:
each frame's cue becomes the a maximizing P(a) - c * P(rest) * [a is a movement] over
all classes (rule C1, as SafetyAgent does), and c is the value with the highest all-frame
accuracy among those that cut the unsafe rate (movement cue on a true-rest frame, per
frame) by at least `UNSAFE_REDUCTION` relative to the model. The replay applies only
this decision, not R2-escalate, so it approximates the full safety agent.

Last, the evaluation agent's linear UE-FMA calibration mu_cal = clip(a + b * mu, 0, 66)
is fitted by least squares of the true UE-FMA on the model's predicted UE-FMA, one point
per validation exercise (the mean of its usable window scores). b is clamped to
[1, 6] (a is then refitted for that b); with fewer than 3 subjects the identity
(a = 0, b = 1) is kept.

The chosen planning settings, `rest_cost`, `ue_calib_a` and `ue_calib_b` are written to
--output (tuned_config.json).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from multi_agents.agents.planning import blended_transition, filter_window  # noqa: E402
from multi_agents.data import open_source  # noqa: E402
from multi_agents.priors import PopulationPriors, resolve_priors  # noqa: E402
from multi_agents.settings import (  # noqa: E402
    ACTIVE_IDS,
    NUM_PRIMITIVES,
    REST_ID,
    LOG_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TUNING,
    MASConfig,
)
from multi_agents.system import make_backend, prepare_model  # noqa: E402
from multi_agents.train_bridge import start_log, train_config  # noqa: E402

# Grids and thresholds: `mas.tuning` in config.yaml.
GRID = {key: tuple(float(value) for value in values) for key, values in TUNING["grid"].items()}
REST_COSTS = tuple(float(value) for value in TUNING["rest_costs"])
UE_CALIB_SLOPE_RANGE = tuple(float(value) for value in TUNING["ue_calibration_slope_range"])
UE_CALIB_MIN_SUBJECTS = int(TUNING["ue_calibration_min_subjects"])
TOLERANCE = float(TUNING["accuracy_tolerance"])
UNSAFE_REDUCTION = float(TUNING["unsafe_reduction"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="Default: paths.checkpoint in config.yaml")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path,
                        default=train_config.experiment_path(OUTPUT_DIR) / "tuned_config.json")
    return parser.parse_args()


def grid_configs(base: MASConfig):
    """Every grid point; settings the plan cannot depend on are not repeated."""
    seen = set()
    for values in itertools.product(*GRID.values()):
        setting = dict(zip(GRID, values))
        if setting["transition_strength"] == 0.0:
            # Without the transition term the plan is the model's arg-max whatever the rest.
            setting["emission_temperature"] = 1.0
            setting["population_prior_strength"] = base.population_prior_strength
        key = tuple(setting.values())
        if key not in seen:
            seen.add(key)
            yield setting, replace(base, **setting)


def cost_decision(posteriors: np.ndarray, cost: float) -> np.ndarray:
    """Per frame, the a maximizing P(a) - cost * P(rest) * [a is a movement] (rule C1)."""
    active = np.isin(np.arange(NUM_PRIMITIVES), ACTIVE_IDS)
    utility = posteriors - cost * posteriors[:, [REST_ID]] * active
    return utility.argmax(axis=1)


def simulate(config: MASConfig, priors: PopulationPriors, recorded: dict, cost: float | None = None) -> np.ndarray:
    """Released labels [N, 5] for every window, mirroring the graph's memory updates.

    With `cost`, each frame takes the safety cost decision (see `cost_decision`);
    otherwise the plan's arg-max.
    """
    minimum = config.patient_count_min_confidence
    planned = []
    for exercises in recorded.values():
        counts = np.zeros((NUM_PRIMITIVES, NUM_PRIMITIVES))
        for _task, result in exercises:
            carry, last = None, None
            for window in range(result.n_windows):
                probs = result.probs[window]
                usable = bool(result.window_ok[window])
                transition = blended_transition(priors, counts, config.population_prior_strength)
                posteriors, carry = filter_window(probs, carry, transition, config, usable)
                if cost is None:
                    labels = posteriors.argmax(axis=1)
                else:
                    labels = cost_decision(posteriors, cost)
                planned.append(labels)
                # deliver: confident released cues feed the patient's statistics (the
                # transition from the previous window's last cue is counted here, once)
                previous = last
                for cue, confidence in zip(labels, posteriors[np.arange(len(labels)), labels]):
                    if confidence >= minimum and previous is not None:
                        counts[previous, cue] += 1.0
                    previous = cue
                last = int(labels[-1])
    return np.asarray(planned)


def fit_uefma_calibration(recorded: dict, truth: dict[str, float]) -> tuple[float, float, pd.DataFrame]:
    """(a, b, points) of true UE-FMA ~ a + b * the model's exercise-mean UE-FMA.

    One point per exercise with at least one usable window. b is clamped to
    UE_CALIB_SLOPE_RANGE and a is the least-squares intercept for that b; with fewer
    than UE_CALIB_MIN_SUBJECTS subjects the identity (0, 1) is returned. `points` holds
    every point used plus the fit (fitted_a, fitted_b, slope_unclamped, n_subjects).
    """
    rows = []
    for subject, exercises in recorded.items():
        for exercise_index, (task, result) in enumerate(exercises):
            scores = np.asarray(result.ue_fma, dtype=np.float64)[np.asarray(result.window_ok, dtype=bool)]
            scores = scores[np.isfinite(scores)]
            if scores.size:
                rows.append(
                    {
                        "subject_id": subject,
                        "exercise_index": exercise_index,
                        "task_name": task,
                        "predicted_mean": float(scores.mean()),
                        "true_ue_fma": float(truth[subject]),
                    }
                )
    points = pd.DataFrame(rows, columns=["subject_id", "exercise_index", "task_name", "predicted_mean", "true_ue_fma"])
    n_subjects = int(points["subject_id"].nunique())
    a, b, slope = 0.0, 1.0, np.nan
    if n_subjects >= UE_CALIB_MIN_SUBJECTS:
        x, y = points["predicted_mean"].to_numpy(), points["true_ue_fma"].to_numpy()
        spread = float(((x - x.mean()) ** 2).sum())
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / spread) if spread > 0 else 1.0
        b = float(np.clip(slope, *UE_CALIB_SLOPE_RANGE))
        a = float(y.mean() - b * x.mean())
    return a, b, points.assign(fitted_a=a, fitted_b=b, slope_unclamped=slope, n_subjects=n_subjects)


def score(planned: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    valid = labels != train_config.MASK_LABEL_ID
    classes = list(range(NUM_PRIMITIVES))
    t4 = valid[:, -1]
    unsafe = np.isin(planned, ACTIVE_IDS) & (labels == REST_ID)
    return {
        "unsafe_rate": float(unsafe[valid].mean()),
        "accuracy_all": float((planned[valid] == labels[valid]).mean()),
        "accuracy_future_primitive": float((planned[t4, -1] == labels[t4, -1]).mean()),
        "macro_f1_all": float(f1_score(labels[valid], planned[valid], labels=classes, average="macro", zero_division=0)),
        "macro_f1_future_primitive": float(
            f1_score(labels[t4, -1], planned[t4, -1], labels=classes, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    args = parse_args()
    start_log(LOG_DIR, f"tune_{args.output.parent.name}")
    base = MASConfig()
    setup = prepare_model(args.checkpoint, args.base_model, not args.no_download)
    source = open_source(
        split="validation", raw_dir=args.raw_dir, conditioning=setup.conditioning, normalizer=setup.normalizer
    )
    priors = resolve_priors(args.raw_dir)
    overlap = set(priors.subjects) & set(source.subjects)
    if overlap:
        raise SystemExit(f"Priors were fitted on subjects being tuned on: {sorted(overlap)}")

    backend = make_backend(source, setup, args.device, args.batch_size)
    recorded, labels = {}, []
    for subject in source.subjects:
        exercises = source.exercises(subject)
        recorded[subject] = [
            (e.task_name, backend.run(e.key)) for e in tqdm(exercises, desc=f"model  {subject}", unit="exercise", dynamic_ncols=True)
        ]
        labels.extend(source.labels(e) for e in exercises)
        print(f"  {subject}: {len(exercises)} exercises")
    labels = np.concatenate(labels)
    model = np.concatenate([r.probs.argmax(axis=2) for rs in recorded.values() for _, r in rs])
    baseline = score(model, labels)
    print(f"Model alone: {baseline}")

    rows = []
    for setting, config in tqdm(list(grid_configs(base)), desc="grid", unit="setting", dynamic_ncols=True):
        rows.append({**setting, **score(simulate(config, priors, recorded), labels)})
    grid = pd.DataFrame(rows).sort_values("accuracy_future_primitive", ascending=False)
    eligible = grid[grid["accuracy_all"] >= baseline["accuracy_all"] - TOLERANCE]
    best = eligible.iloc[0] if len(eligible) else None
    if best is None or best["accuracy_future_primitive"] <= baseline["accuracy_future_primitive"]:
        chosen = {key: getattr(MASConfig(), key) for key in GRID}
        print("No setting beats the model on the validation subjects; keeping the do-no-harm defaults.")
    else:
        chosen = {key: float(best[key]) for key in GRID}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    planning_config = replace(base, **chosen)
    target = (1.0 - UNSAFE_REDUCTION) * baseline["unsafe_rate"]
    curve = pd.DataFrame(
        [
            {"rest_cost": cost, **score(simulate(planning_config, priors, recorded, cost), labels)}
            for cost in tqdm(REST_COSTS, desc="cost", unit="c", dynamic_ncols=True)
        ]
    )
    curve["meets_target"] = curve["unsafe_rate"] <= target
    if curve["meets_target"].any():
        rest_cost = float(curve[curve["meets_target"]].sort_values("accuracy_all", ascending=False).iloc[0]["rest_cost"])
    else:
        rest_cost = float(curve.sort_values("unsafe_rate").iloc[0]["rest_cost"])
        print(f"WARNING: no rest_cost reaches unsafe rate <= {target:.4f}; using the safest, {rest_cost}")
    chosen["rest_cost"] = rest_cost

    truth = {subject: source.true_uefma(subject) for subject in recorded}
    ue_calib_a, ue_calib_b, calib_points = fit_uefma_calibration(recorded, truth)
    calib_subjects = int(calib_points["subject_id"].nunique())
    if calib_subjects < UE_CALIB_MIN_SUBJECTS:
        print(f"WARNING: UE-FMA calibration needs >= {UE_CALIB_MIN_SUBJECTS} subjects, got {calib_subjects}; keeping a = 0, b = 1")
    else:
        slope = float(calib_points["slope_unclamped"].iloc[0])
        print(
            f"UE-FMA calibration on {calib_subjects} subjects ({len(calib_points)} exercises): "
            f"true ~ {ue_calib_a:.3f} + {ue_calib_b:.3f} * predicted (slope before clamping {slope:.3f})"
        )
    chosen.update(ue_calib_a=ue_calib_a, ue_calib_b=ue_calib_b)

    args.output.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(grid.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + curve.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nChosen: {chosen}\nWrote {args.output}")


if __name__ == "__main__":
    main()
