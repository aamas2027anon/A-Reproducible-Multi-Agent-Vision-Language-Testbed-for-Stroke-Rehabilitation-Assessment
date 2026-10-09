"""Test-style evaluation: the single VLM against the multi-agent system.

    python -m multi_agents.evaluate --split test --mas-config tuned_config.json

Modes:
  vlm              the fine-tuned model alone (the training pipeline's test predictions)
  mas              full agent loop with patient memory (InMemorySaver, isolated run)
  mas_no_memory    ablation: as mas, patient memory reset before every exercise
  mas_no_planning  ablation: as mas, planning passes the model's cues through
  mas_no_safety    ablation: as mas, safety approves every usable window and the
                   coordinator's arbitration uses no rest penalty (rest_cost = 0)

Every MAS mode is the tuned config plus the mode's overrides (`MODE_OVERRIDES`), recorded
with its hash in run_manifest.json. `exercise_limit.csv` re-scores every mode on each
patient's first N exercises (see `history_analysis.py`).

The model runs once per exercise; its outputs are cached on disk, and every mode
replays the same outputs, so differences between modes come only from the agents.

Per mode the output has the training pipeline's layout (`predictions/`, and `final/`
with the position, per-class, event and task tables, confusion matrices and counting),
so training's multi-seed analysis reads it unchanged. Each MAS mode also gets
`mas_analysis/`: what the agents changed relative to the VLM (see `mas_analysis.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from multi_agents import history_analysis, mas_analysis, metrics, plots, sequence_metrics  # noqa: E402
from multi_agents.data import open_source  # noqa: E402
from multi_agents.graph import build_graph  # noqa: E402
from multi_agents.memory import WindowLog, export_patient_memory, make_checkpointer  # noqa: E402
from multi_agents.priors import resolve_priors  # noqa: E402
from multi_agents.session import RehabSession  # noqa: E402
from multi_agents.settings import OUTPUT_DIR, RAW_DATA_DIR, MASConfig  # noqa: E402
from multi_agents.train_bridge import train_config, write_final_artifacts  # noqa: E402
from multi_agents.system import (  # noqa: E402
    build_services,
    log_namespace,
    make_backend,
    patient_seed,
    prepare_model,
)

MODES = ("vlm", "mas", "mas_no_memory", "mas_no_planning", "mas_no_safety")
# Each MAS mode is the tuned config with these changes (mas_no_memory differs by running
# every exercise on a fresh patient thread instead).
MODE_OVERRIDES: dict[str, dict] = {
    "mas_no_planning": {"transition_strength": 0.0, "emission_temperature": 1.0},
    # rest_cost = 0: arbitration takes the higher-belief cue (planning's), with no rest penalty.
    "mas_no_safety": {"safety_enabled": False, "rest_cost": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("test", "validation", "none"), default="test")
    parser.add_argument("--modes", default=",".join(MODES), help=f"Comma list of {MODES}")
    parser.add_argument("--subjects", default=None, help="Comma list overriding the split's subjects")
    parser.add_argument("--order", choices=("session", "protocol", "random"), default="session")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --order random")
    parser.add_argument("--limit-exercises", type=int, default=None, help="Use only the first N exercises per subject")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--mas-config", type=Path, default=None)
    parser.add_argument("--reference-predictions", default=None, help="Training test_combined_window_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timelines-per-subject", type=int, default=1)
    return parser.parse_args()


def vlm_records(backend, exercises_by_subject: dict) -> list[dict]:
    records = []
    for subject_id, exercises in exercises_by_subject.items():
        for ordinal, exercise in enumerate(exercises):
            result = backend.run(exercise.key)
            for window_index in range(result.n_windows):
                report = result.report(window_index)
                labels, confidence = report.predicted_labels, report.confidence
                records.append(
                    {
                        "mode": "vlm",
                        "subject_id": subject_id,
                        "clip_id": exercise.clip_id,
                        "task_name": exercise.task_name,
                        "exercise_index": ordinal,
                        "window_index": window_index,
                        "target_frame_index": report.target_frame_index,
                        "integrity_ok": report.integrity_ok,
                        "model_action": labels,
                        "model_confidence": confidence,
                        "model_p_rest": [row["rest"] for row in report.probs],
                        "model_ue_fma": report.ue_fma,
                        "plan_action": labels,
                        "plan_confidence": confidence,
                        "final_action": labels,
                        "final_confidence": confidence,
                        "status": "approve",
                        "released": True,
                        "selected_agent": "perception",
                        "disagreement": False,
                        "coordination_failure": None,
                        "rule_hits": [],
                        "band": None,
                        "estimated_uefma": None,
                        "current_primitive": labels[0],
                        "next_primitive": labels[-1],
                        "next_confidence": confidence[-1],
                        "feedback": "",
                        "step_ms": result.window(window_index).model_ms,
                    }
                )
    return records


def run_mas_mode(mode, config, setup, source, priors, backend, exercises_by_subject, run_id):
    """Run one MAS mode; return (records, exercise records, graph, threads)."""
    checkpointer, _ = make_checkpointer("inmemory")
    log = WindowLog(log_namespace(config, setup))
    services = build_services(
        config=config, priors=priors, backend=backend, log=log, mode=mode
    )
    graph = build_graph(services, checkpointer)
    exercise_records: dict[str, list[dict]] = {}
    threads: dict[str, str] = {}

    for subject_id, exercises in exercises_by_subject.items():
        seed = patient_seed(source, subject_id, config, setup)
        thread_id = f"{run_id}:{mode}:{subject_id}"
        threads[subject_id] = thread_id
        started = time.perf_counter()
        session = RehabSession(graph, services, thread_id, seed)
        for ordinal, exercise in enumerate(tqdm(exercises, desc=f"{mode} {subject_id}", unit="exercise", dynamic_ncols=True)):
            if mode == "mas_no_memory":
                session = RehabSession(graph, services, f"{run_id}:{mode}:{subject_id}:{exercise.clip_id}", seed)
            first_record = len(log.records)
            exercise_report = session.replay(exercise)
            # With per-exercise threads every exercise is the patient's "first"; use the
            # sequence position so trajectories line up across modes.
            record = {**exercise_report["record"], "exercise_index": ordinal}
            for window in log.records[first_record:]:
                window["exercise_index"] = ordinal
            exercise_records.setdefault(subject_id, []).append(record)
        print(f"  {mode} {subject_id}: {len(exercises)} exercises in {time.perf_counter() - started:.1f}s")

    records = list(log.records)
    for record in records:
        record["mode"] = mode
    return records, exercise_records, graph, threads


def main() -> None:
    args = parse_args()
    requested = {m.strip() for m in args.modes.split(",") if m.strip()}
    unknown = requested - set(MODES)
    if unknown:
        raise SystemExit(f"Unknown modes {sorted(unknown)}; choose from {MODES}")
    # Fixed order: vlm first (every MAS mode is analysed against it).
    modes = [m for m in MODES if m in requested]
    config = MASConfig.from_json(args.mas_config)
    mode_configs = {mode: replace(config, **MODE_OVERRIDES.get(mode, {})) for mode in MODES}
    setup = prepare_model(args.checkpoint, args.base_model, not args.no_download)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output = args.output_dir or OUTPUT_DIR / f"eval_{args.split}_{run_id}"
    output.mkdir(parents=True, exist_ok=True)
    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None

    print(f"Checkpoint {setup.checkpoint.name} ({setup.conditioning}); output {output}")
    source = open_source(
        split=args.split,
        raw_dir=args.raw_dir,
        conditioning=setup.conditioning,
        normalizer=setup.normalizer,
        subjects=subjects,
    )
    priors = resolve_priors(args.raw_dir)
    exercises_by_subject = {
        subject: source.exercises(subject, args.order, args.seed)[: args.limit_exercises]
        for subject in source.subjects
    }

    backend = make_backend(source, setup, args.device, args.batch_size)
    print("Running the movement model (cached per exercise) ...")
    inference_ms: dict[str, float] = {}
    for subject, exercises in exercises_by_subject.items():
        started = time.perf_counter()
        for exercise in tqdm(exercises, desc=f"model  {subject}", unit="exercise", dynamic_ncols=True):
            inference_ms[exercise.key] = backend.run(exercise.key).latency_ms
        print(f"  {subject}: {len(exercises)} exercises in {time.perf_counter() - started:.1f}s")

    split = args.split
    summaries, per_subject, primitives_by_mode, exercises_by_mode, records_by_mode = [], {}, {}, {}, {}
    sequence_by_mode: dict[str, pd.DataFrame] = {}
    for mode in modes:
        print(f"Mode {mode} ...")
        if mode == "vlm":
            records, exercise_records = vlm_records(backend, exercises_by_subject), None
        else:
            records, exercise_records, graph, threads = run_mas_mode(
                mode, mode_configs[mode], setup, source, priors, backend, exercises_by_subject, run_id
            )
            export_patient_memory(graph, threads, output / mode / "patient_memory.jsonl")
        frame = pd.DataFrame(records)
        primitives = metrics.primitive_frame(frame, source, mode, split)
        exercises = metrics.exercise_frame(frame, primitives, source, mode, exercise_records, mode_configs[mode])
        uefma = metrics.uefma_subject_frame(frame, exercises, source, mode, split, mode_configs[mode])
        combined = metrics.combined_frame(primitives, frame, exercises, mode, mode_configs[mode])
        sequence_clips = sequence_metrics.clip_sequence_frame(primitives)

        mode_dir = output / mode
        prediction_dir = mode_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        primitives[list(metrics.PRIMITIVE_EXPORT_COLUMNS)].to_csv(prediction_dir / f"{split}_primitive_predictions.csv", index=False)
        uefma[list(metrics.UEFMA_EXPORT_COLUMNS)].to_csv(prediction_dir / f"{split}_ue_fma_subject_predictions.csv", index=False)
        combined.to_csv(prediction_dir / f"{split}_combined_window_predictions.csv", index=False)
        exercises.to_csv(mode_dir / "exercise_trajectory.csv", index=False)
        sequence_clips.to_csv(mode_dir / "sequence_metrics_clips.csv", index=False, float_format="%.6f")
        frame.to_json(mode_dir / "window_decisions.jsonl", orient="records", lines=True)
        write_final_artifacts(combined, mode_dir, mode, split)

        summary = metrics.summary_metrics(
            mode=mode,
            primitives=primitives,
            records=frame,
            exercises=exercises,
            uefma_subjects=uefma,
            exercise_records=exercise_records,
            inference_ms=inference_ms,
            config=mode_configs[mode],
            priors=priors,
            sequence_clips=sequence_clips,
        )
        summaries.append(summary)
        per_subject[mode] = metrics.per_subject_metrics(primitives, exercises)
        primitives_by_mode[mode], exercises_by_mode[mode], records_by_mode[mode] = primitives, exercises, frame
        sequence_by_mode[mode] = sequence_clips
        if mode != "vlm" and "vlm" in primitives_by_mode:
            mas_analysis.analyse_mode(
                mode=mode,
                records=frame,
                primitives=primitives,
                vlm_primitives=primitives_by_mode["vlm"],
                summary=summary,
                vlm_summary=summaries[0],
                output_dir=mode_dir / "mas_analysis",
            )

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(output / "comparison.csv", index=False)
    history_analysis.exercise_limit_frame(primitives_by_mode, exercises_by_mode, config).to_csv(
        output / "exercise_limit.csv", index=False
    )
    paired = metrics.paired_comparison(per_subject)
    paired.to_csv(output / "paired_comparison.csv", index=False)
    if args.reference_predictions and "vlm" in primitives_by_mode:
        agreement = metrics.reference_agreement(primitives_by_mode["vlm"], args.reference_predictions)
        (output / "reference_agreement.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
        print(f"VLM vs training predictions: {agreement}")

    figures = output / "figures"
    plots.comparison_bars(comparison, figures / "comparison_bars.png")
    plots.safety_tradeoff(comparison, figures / "safety_tradeoff.png")
    plots.latency_breakdown(comparison, figures / "latency_per_window.png")
    plots.risk_coverage(primitives_by_mode, figures / "risk_coverage.png")
    plots.adaptation_curve(exercises_by_mode, figures / "adaptation_curve.png")
    plots.sequence_metrics(sequence_by_mode, figures / "sequence_metrics.png")
    plots.sequence_metrics_vs_vlm(sequence_by_mode, figures / "sequence_metrics_vs_vlm.png")
    mas_mode = next((m for m in ("mas", "mas_no_memory") if m in exercises_by_mode), None)
    if mas_mode is not None:
        for subject in exercises_by_mode[mas_mode]["subject_id"].unique():
            plots.patient_trajectory(exercises_by_mode, subject, figures / "trajectories" / f"{subject}.png", mas_mode)
            ranked = exercises_by_mode[mas_mode][exercises_by_mode[mas_mode]["subject_id"] == subject]
            ranked = ranked.assign(interventions=ranked["revise"] + ranked["escalate"] + ranked["block"])
            for clip_id in ranked.nlargest(args.timelines_per_subject, "interventions")["clip_id"]:
                plots.decision_timeline(
                    records_by_mode[mas_mode],
                    primitives_by_mode[mas_mode],
                    clip_id,
                    figures / "timelines" / f"{subject}_{clip_id}.png",
                    subject_id=subject,
                )

    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "split": split,
                "modes": modes,
                "subjects": list(exercises_by_subject),
                "exercises": {s: len(e) for s, e in exercises_by_subject.items()},
                "order": args.order,
                "checkpoint": train_config.portable_path(setup.checkpoint),
                "checkpoint_identity": setup.identity,
                "conditioning": setup.conditioning,
                "base_model": train_config.portable_path(setup.base_model),
                "source": "raw",
                "mas_config": config.to_dict(),
                "mas_config_hash": config.config_hash(),
                # UE-FMA calibration mu_cal = clip(a + b * mu, 0, 66), applied in every mode's UE-FMA metrics.
                "ue_calibration": {"a": config.ue_calib_a, "b": config.ue_calib_b},
                "mode_overrides": {mode: MODE_OVERRIDES.get(mode, {}) for mode in modes if mode != "vlm"},
                "mode_config_hashes": {mode: mode_configs[mode].config_hash() for mode in modes if mode != "vlm"},
                "prior_subjects": list(priors.subjects),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    columns = [
        "mode",
        "accuracy_future_primitive",
        "accuracy_all",
        "macro_f1_all",
        "count_mae",
        "ue_fma_exercise_mae",
        "unsafe_recommendation_rate",
        "unnecessary_escalation_rate",
        "coverage_windows",
    ]
    print("\n" + comparison[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nResults in {output}")


if __name__ == "__main__":
    main()
