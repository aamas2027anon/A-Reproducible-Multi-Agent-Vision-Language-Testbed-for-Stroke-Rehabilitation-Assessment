"""Evaluate the VLM and the multi-agent system on the test subjects.

    python -m multi_agents.evaluate --checkpoint <best.ckpt> --mas-config tuned_config.json

Modes:
  vlm              the fine-tuned model alone
  mas              full agent loop with patient memory
  mas_no_memory    ablation: as mas, patient memory reset before every exercise
  mas_no_planning  ablation: as mas, planning passes the model's cues through
  mas_no_safety    ablation: as mas, safety approves every usable window and the
                   coordinator's arbitration uses no rest penalty (rest_cost = 0)

Every MAS mode is the tuned config plus the mode's overrides (`MODE_OVERRIDES`). The
model runs once per exercise; its outputs are cached on disk and every mode replays
them, so differences between modes come only from the agents. Each patient's exercises
run in session order through one memory thread. Written to --output-dir:

  metrics.csv              one row per mode (see metrics.summary_metrics)
  change_attribution.csv   per MAS mode, the frames planning and safety changed
  count_errors.csv         per mode, subject and primitive: actual and predicted counts
  primitive_predictions.csv scored frames and decisions for every mode
  exercise_predictions.csv running and isolated exercise UE-FMA estimates
  subject_predictions.csv  final subject UE-FMA estimates
  run_manifest.json        checkpoint, configs and subjects
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from multi_agents import metrics  # noqa: E402
from multi_agents.data import open_source  # noqa: E402
from multi_agents.graph import build_graph  # noqa: E402
from multi_agents.memory import WindowLog, make_checkpointer  # noqa: E402
from multi_agents.priors import resolve_priors  # noqa: E402
from multi_agents.session import RehabSession  # noqa: E402
from multi_agents.settings import LOG_DIR, OUTPUT_DIR, RAW_DATA_DIR, MASConfig  # noqa: E402
from multi_agents.system import (  # noqa: E402
    build_services,
    log_namespace,
    make_backend,
    patient_seed,
    prepare_model,
)
from multi_agents.train_bridge import start_log, train_config  # noqa: E402

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
    parser.add_argument("--checkpoint", default=None, help="Default: paths.checkpoint in config.yaml")
    parser.add_argument("--mas-config", type=Path, default=None, help="tuned_config.json from multi_agents.tune")
    parser.add_argument("--output-dir", type=Path,
                        default=train_config.experiment_path(OUTPUT_DIR) / "eval_test")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def vlm_records(backend, exercises_by_subject: dict) -> list[dict]:
    """The model alone: every window's argmax cue, all released."""
    records = []
    for subject_id, exercises in exercises_by_subject.items():
        for ordinal, exercise in enumerate(exercises):
            result = backend.run(exercise.key)
            for window_index in range(result.n_windows):
                report = result.report(window_index)
                labels = report.predicted_labels
                records.append(
                    {
                        "subject_id": subject_id,
                        "clip_id": exercise.clip_id,
                        "exercise_index": ordinal,
                        "window_index": window_index,
                        "target_frame_index": report.target_frame_index,
                        "model_action": labels,
                        "model_ue_fma": report.ue_fma,
                        "plan_action": labels,
                        "final_action": labels,
                        "released": True,
                        "coordination_failure": None,
                    }
                )
    return records


def run_mas_mode(mode, config, setup, source, priors, backend, exercises_by_subject, run_id):
    """Run one MAS mode; return (window records, exercise records per subject)."""
    checkpointer, _ = make_checkpointer("inmemory")
    log = WindowLog(log_namespace(config, setup))
    services = build_services(config=config, priors=priors, backend=backend, log=log, mode=mode)
    graph = build_graph(services, checkpointer)
    exercise_records: dict[str, list[dict]] = {}
    for subject_id, exercises in exercises_by_subject.items():
        seed = patient_seed(source, subject_id, config, setup)
        session = RehabSession(graph, services, f"{run_id}:{mode}:{subject_id}", seed)
        for ordinal, exercise in enumerate(tqdm(exercises, desc=f"{mode} {subject_id}", unit="exercise", dynamic_ncols=True)):
            if mode == "mas_no_memory":
                session = RehabSession(graph, services, f"{run_id}:{mode}:{subject_id}:{exercise.clip_id}", seed)
            first_record = len(log.records)
            record = {**session.replay(exercise)["record"], "exercise_index": ordinal}
            # With per-exercise threads every exercise is the patient's "first"; use the
            # sequence position so exercises line up across modes.
            for window in log.records[first_record:]:
                window["exercise_index"] = ordinal
            exercise_records.setdefault(subject_id, []).append(record)
    return list(log.records), exercise_records


def main() -> None:
    args = parse_args()
    start_log(LOG_DIR, f"evaluate_{args.output_dir.parent.name}")
    config = MASConfig.from_json(args.mas_config)
    mode_configs = {mode: replace(config, **MODE_OVERRIDES.get(mode, {})) for mode in MODES}
    setup = prepare_model(args.checkpoint, args.base_model, not args.no_download)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint {setup.checkpoint.name} ({setup.conditioning}); output {output}")
    source = open_source(split="test", raw_dir=args.raw_dir, conditioning=setup.conditioning, normalizer=setup.normalizer)
    priors = resolve_priors(args.raw_dir)
    exercises_by_subject = {subject: source.exercises(subject) for subject in source.subjects}

    backend = make_backend(source, setup, args.device, args.batch_size)
    print("Running the movement model (cached per exercise) ...")
    for subject, exercises in exercises_by_subject.items():
        for exercise in tqdm(exercises, desc=f"model  {subject}", unit="exercise", dynamic_ncols=True):
            backend.run(exercise.key)

    rows, attribution, counts = [], [], []
    for mode in MODES:
        print(f"Mode {mode} ...")
        if mode == "vlm":
            records, exercise_records = vlm_records(backend, exercises_by_subject), None
        else:
            records, exercise_records = run_mas_mode(
                mode, mode_configs[mode], setup, source, priors, backend, exercises_by_subject, run_id
            )
        frame = pd.DataFrame(records)
        primitives = metrics.primitive_frame(frame, source)
        exercises = metrics.exercise_frame(frame, source, exercise_records, mode_configs[mode])
        uefma = metrics.uefma_subject_frame(frame, exercises, source, mode, mode_configs[mode])
        mode_counts = metrics.subject_primitive_counts(primitives)
        for name, evidence in (
            ("primitive_predictions.csv", primitives),
            ("exercise_predictions.csv", exercises),
            ("subject_predictions.csv", uefma),
        ):
            evidence.assign(mode=mode).to_csv(
                output / name, index=False, mode="w" if mode == "vlm" else "a", header=mode == "vlm"
            )
        rows.append(
            metrics.summary_metrics(
                mode=mode,
                primitives=primitives,
                records=frame,
                exercises=exercises,
                uefma_subjects=uefma,
                counts=mode_counts,
                config=mode_configs[mode],
                priors=priors,
            )
        )
        counts.append(mode_counts.assign(mode=mode))
        if mode != "vlm":
            attribution += [{"mode": mode, **row} for row in metrics.change_attribution(primitives)]

    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(attribution).to_csv(output / "change_attribution.csv", index=False)
    pd.concat(counts, ignore_index=True).to_csv(output / "count_errors.csv", index=False)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "subjects": list(exercises_by_subject),
                "exercises": {s: len(e) for s, e in exercises_by_subject.items()},
                "checkpoint": train_config.portable_path(setup.checkpoint),
                "checkpoint_identity": setup.identity,
                "conditioning": setup.conditioning,
                "base_model": train_config.portable_path(setup.base_model),
                "mas_config": config.to_dict(),
                "mode_overrides": {mode: MODE_OVERRIDES.get(mode, {}) for mode in MODES if mode != "vlm"},
                "prior_subjects": list(priors.subjects),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    columns = ["mode", "accuracy", "accuracy_h4", "count_mae", "ue_fma_exercise_mae", "unsafe_rate", "coverage"]
    print("\n" + table[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nResults in {output}")


if __name__ == "__main__":
    main()
