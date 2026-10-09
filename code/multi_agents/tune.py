"""Tune the planning agent, the safety cost and the UE-FMA calibration on the validation subjects (never on test).

    python -m multi_agents.tune --split validation
    python -m multi_agents.evaluate --split test --mas-config multi_agent_outputs/tuned_config.json

The model runs once per validation exercise (cached, like the evaluation). The grid is
then scored on CPU by replaying the planning filter and the patient-memory updates
exactly as the graph applies them, patient by patient in session order, without the
safety and coordination steps (they only act on top of the plan).

Selection: the highest next-primitive (frame t+4) accuracy, among settings whose
accuracy over all five frames is no worse than the model's by more than `--tolerance`.
The defaults (transition_strength = 0) reproduce the model, so a
setting is only chosen if it beats it.

Then, with those planning settings, the safety agent's cost c (`rest_cost`) is chosen:
each frame's cue becomes the a maximizing P(a) - c * P(rest) * [a is a movement] over
all classes (rule C1, as SafetyAgent does), and c is the value with the highest all-frame
accuracy among those that cut the unsafe rate (movement cue on a true-rest frame, per
frame) by at least `--unsafe-reduction` relative to the model. The replay applies only
this decision, not R2-escalate, so it approximates the full safety agent.
The curve over c is saved as cost_curve.csv / cost_curve.png.

Last, the evaluation agent's linear UE-FMA calibration mu_cal = clip(a + b * mu, 0, 66)
is fitted by least squares of the true UE-FMA on the model's predicted UE-FMA, one point
per validation exercise (the mean of its usable window scores). b is clamped to
[1, 6] (a is then refitted for that b); with fewer than 3 subjects the identity
(a = 0, b = 1) is kept. `ue_calib_a` and `ue_calib_b` are saved in tuned_config.json; the
points and the fit (with the slope before clamping) go to ue_calibration.csv.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
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
    OUTPUT_DIR,
    RAW_DATA_DIR,
    MASConfig,
)
from multi_agents.system import make_backend, prepare_model  # noqa: E402
from multi_agents.train_bridge import train_config  # noqa: E402

GRID = {
    "transition_strength": (0.0, 0.25, 0.5, 0.75, 1.0),
    "emission_temperature": (0.7, 1.0, 1.5),
    "population_prior_strength": (500.0, 2000.0, 8000.0),
}
REST_COSTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
UE_CALIB_SLOPE_RANGE = (1.0, 6.0)
UE_CALIB_MIN_SUBJECTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("validation", "test", "none"), default="validation")
    parser.add_argument("--subjects", default=None, help="Comma list overriding the split's subjects")
    parser.add_argument("--order", choices=("session", "protocol", "random"), default="session")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Allowed drop in all-frame accuracy vs the model")
    parser.add_argument(
        "--unsafe-reduction",
        type=float,
        default=0.18,
        help="Safety cost: required relative cut in the unsafe rate vs the model (0.18 = what the former revise-to-rest rules achieved)",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--mas-config", type=Path, default=None, help="Base config the tuned keys are added to")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "tuned_config.json")
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
    every point used plus the fit (fitted_a, fitted_b, slope_unclamped, n_subjects), for
    ue_calibration.csv.
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


def plot_cost_curve(curve: pd.DataFrame, baseline: dict, target: float, chosen: float, path: Path) -> None:
    """Accuracy against unsafe rate for each rest_cost, with the model and the target."""
    from multi_agents import plots

    plots._style()
    figure, axis = plots.plt.subplots(figsize=(7, 5))
    axis.plot(curve["unsafe_rate"], curve["accuracy_all"], "-o", color=plots.mode_color("mas"), label="safety cost")
    for row in curve.itertuples(index=False):
        axis.annotate(f"c={row.rest_cost:g}", (row.unsafe_rate, row.accuracy_all), fontsize=7,
                      xytext=(4, 3), textcoords="offset points", color=plots.INK_SECONDARY)
    picked = curve[curve["rest_cost"] == chosen].iloc[0]
    axis.scatter([picked["unsafe_rate"]], [picked["accuracy_all"]], s=140, facecolors="none",
                 edgecolors=plots.INK, linewidths=1.5, zorder=4, label=f"chosen c={chosen:g}")
    axis.scatter([baseline["unsafe_rate"]], [baseline["accuracy_all"]], s=60, color=plots.mode_color("vlm"), zorder=4, label="model alone")
    axis.axvline(target, color=plots.INK_SECONDARY, linestyle="--", linewidth=1, label="unsafe-rate target")
    axis.set(xlabel="Unsafe rate (movement cue on a true-rest frame, per frame)", ylabel="Accuracy, all frames",
             title="Safety cost on the validation subjects")
    axis.legend(fontsize=8)
    figure.tight_layout()
    plots._save(figure, path)


def main() -> None:
    args = parse_args()
    base = MASConfig.from_json(args.mas_config)
    setup = prepare_model(args.checkpoint, args.base_model, not args.no_download)
    source = open_source(
        split=args.split,
        raw_dir=args.raw_dir,
        conditioning=setup.conditioning,
        normalizer=setup.normalizer,
        subjects=[s.strip() for s in args.subjects.split(",")] if args.subjects else None,
    )
    priors = resolve_priors(args.raw_dir)
    overlap = set(priors.subjects) & set(source.subjects)
    if overlap:
        raise SystemExit(f"Priors were fitted on subjects being tuned on: {sorted(overlap)}")

    backend = make_backend(source, setup, args.device, args.batch_size)
    recorded, labels = {}, []
    for subject in source.subjects:
        exercises = source.exercises(subject, args.order)
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
    eligible = grid[grid["accuracy_all"] >= baseline["accuracy_all"] - args.tolerance]
    best = eligible.iloc[0] if len(eligible) else None
    if best is None or best["accuracy_future_primitive"] <= baseline["accuracy_future_primitive"]:
        chosen = {key: getattr(MASConfig(), key) for key in GRID}
        print("No setting beats the model on the validation subjects; keeping the do-no-harm defaults.")
    else:
        chosen = {key: float(best[key]) for key in GRID}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    planning_config = replace(base, **chosen)
    target = (1.0 - args.unsafe_reduction) * baseline["unsafe_rate"]
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
    curve.assign(target_unsafe_rate=target, chosen=curve["rest_cost"] == rest_cost).to_csv(
        args.output.with_name("cost_curve.csv"), index=False
    )
    plot_cost_curve(curve, baseline, target, rest_cost, args.output.with_name("cost_curve.png"))

    truth = {subject: source.true_uefma(subject) for subject in recorded}
    ue_calib_a, ue_calib_b, calib_points = fit_uefma_calibration(recorded, truth)
    calib_subjects = int(calib_points["subject_id"].nunique())
    calib_points.to_csv(args.output.with_name("ue_calibration.csv"), index=False)
    if calib_subjects < UE_CALIB_MIN_SUBJECTS:
        print(f"WARNING: UE-FMA calibration needs >= {UE_CALIB_MIN_SUBJECTS} subjects, got {calib_subjects}; keeping a = 0, b = 1")
    else:
        slope = float(calib_points["slope_unclamped"].iloc[0])
        print(
            f"UE-FMA calibration on {calib_subjects} subjects ({len(calib_points)} exercises): "
            f"true ~ {ue_calib_a:.3f} + {ue_calib_b:.3f} * predicted (slope before clamping {slope:.3f})"
        )
    chosen.update(ue_calib_a=ue_calib_a, ue_calib_b=ue_calib_b)

    # Keep any overrides the tuning started from, then the tuned planning keys, rest_cost
    # and the UE-FMA calibration.
    overrides = json.loads(args.mas_config.read_text(encoding="utf-8")) if args.mas_config else {}
    tuned = {**overrides, **chosen}
    args.output.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    grid_path = args.output.with_name("tuning_grid.csv")
    grid.assign(baseline_accuracy_all=baseline["accuracy_all"], baseline_accuracy_future_primitive=baseline["accuracy_future_primitive"]).to_csv(
        grid_path, index=False
    )
    print(grid.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + curve.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nChosen: {chosen}\nWrote {args.output}, {grid_path}, cost_curve.csv/.png and ue_calibration.csv")


if __name__ == "__main__":
    main()
