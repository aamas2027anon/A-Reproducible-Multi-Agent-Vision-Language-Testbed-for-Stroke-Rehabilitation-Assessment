"""Real-time demonstration: stream one exercise through the multi-agent system.

    python -m multi_agents.demo --sample-id s04_drinking3_1 --replay-history
    python -m multi_agents.demo --sample-id s04_drinking3_1 --fast
    python -m multi_agents.demo --scenario sensor_dropout
    python -m multi_agents.demo --list-scenarios

The target clip is fed frame by frame at the feature frame rate (15 fps by default,
`[stream] fps` in project_config.toml), as a live sensor would deliver it. After 116
frames of warm-up, every 5th frame completes a window and the agents return a
recommendation: the current primitive and the future primitive to follow (4 frames, 0.27 s
ahead). `--replay-history` first runs the patient's earlier exercises from their
recorded model outputs, so the live exercise is decided with that history.

Patient history is kept in a sqlite checkpointer by default, so running the demo again
continues the same patient's record.
Scenarios use their own threads and start fresh every time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from multi_agents.data import open_source  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from multi_agents.graph import build_graph  # noqa: E402
from multi_agents.memory import WindowLog, delete_thread, load_patient, make_checkpointer, parse_memory_spec  # noqa: E402
from multi_agents.priors import resolve_priors  # noqa: E402
from multi_agents.scenarios import SCENARIOS, Scenario  # noqa: E402
from multi_agents.session import RehabSession  # noqa: E402
from multi_agents.settings import (  # noqa: E402
    DEFAULT_SQLITE_PATH,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    STREAM_FPS,
    MASConfig,
)
from multi_agents.train_bridge import train_config  # noqa: E402
from multi_agents.system import (  # noqa: E402
    build_services,
    log_namespace,
    make_backend,
    patient_seed,
    prepare_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_argument_group("what to run")
    target.add_argument("--sample-id", help="Clip id, e.g. s04_drinking3_1")
    target.add_argument("--scenario", choices=sorted(SCENARIOS), help="Run a named scenario instead")
    target.add_argument("--list-scenarios", action="store_true")
    target.add_argument("--replay-history", action="store_true", help="Run the patient's earlier exercises first")
    target.add_argument("--max-history", type=int, default=None, help="Replay at most this many earlier exercises")
    target.add_argument("--order", choices=("session", "protocol", "random"), default="session")
    target.add_argument("--fast", action="store_true", help="Stream without real-time pacing (same outputs)")
    target.add_argument("--fps", type=float, default=None, help="Stream frame rate (default: [stream] fps in project_config.toml)")
    target.add_argument(
        "--print",
        dest="print_mode",
        choices=("every", "changes"),
        default="every",
        help="Print every recommendation, or only when the cue or status changes",
    )
    target.add_argument(
        "--record-repeat",
        action="store_true",
        help="Append a clip already in the patient's history as a new exercise (default: preview only)",
    )

    model = parser.add_argument_group("models")
    model.add_argument("--checkpoint", default=None, help="Trained .ckpt (default: `checkpoint` in project_config.toml, else the only one in checkpoints_dir)")
    model.add_argument("--base-model", default=None, help="Qwen3.5-0.8B-Base directory (downloaded if missing)")
    model.add_argument("--no-download", action="store_true", help="Fail instead of downloading missing models")
    model.add_argument("--device", default=None)
    model.add_argument("--batch-size", type=int, default=16)

    data = parser.add_argument_group("data and memory")
    data.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    data.add_argument("--memory", default=f"sqlite:{DEFAULT_SQLITE_PATH}", help="'inmemory' or 'sqlite:<path>'")
    data.add_argument("--mas-config", type=Path, default=None, help="JSON overrides for MASConfig")
    data.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "demo")
    return parser.parse_args()


def resolve_target(args: argparse.Namespace) -> tuple[Scenario | None, str, str, str | None, bool, int | None]:
    if args.scenario:
        scenario = SCENARIOS[args.scenario]
        return scenario, scenario.subject_id, scenario.target_clip, scenario.split, scenario.replay_history, scenario.max_history
    if not args.sample_id:
        raise SystemExit("Pass --sample-id or --scenario (see --list-scenarios)")
    subject = args.sample_id.split("_", 1)[0]
    return None, subject, args.sample_id, None, args.replay_history, args.max_history


def main() -> None:
    args = parse_args()
    if args.list_scenarios:
        for scenario in SCENARIOS.values():
            print(f"{scenario.name:24s} {scenario.description}")
        return
    scenario, subject, clip_id, split, replay, max_history = resolve_target(args)
    config = MASConfig.from_json(args.mas_config)
    setup = prepare_model(args.checkpoint, args.base_model, not args.no_download)
    perturbations = {clip_id: scenario.perturbation} if scenario and scenario.perturbation else None

    source = open_source(
        split=split or "test",
        raw_dir=args.raw_dir,
        conditioning=setup.conditioning,
        normalizer=setup.normalizer,
        subjects=[subject],
        clip_ids=None,  # all of the patient's clips, so stored history can be scored for the figures
        perturbations=perturbations,
    )
    priors = resolve_priors(args.raw_dir)

    memory_spec = "inmemory" if scenario else args.memory
    checkpointer, sqlite_path = make_checkpointer(memory_spec)
    log = WindowLog(log_namespace(config, setup), None if scenario else sqlite_path)
    backend = make_backend(source, setup, args.device, args.batch_size, use_cache=perturbations is None)
    services = build_services(config=config, priors=priors, backend=backend, log=log, mode="mas")
    graph = build_graph(services, checkpointer)
    thread_id = f"scenario:{scenario.name}:{subject}" if scenario else subject
    if scenario:
        delete_thread(checkpointer, thread_id)
    seed = patient_seed(source, subject, config, setup)

    stored = load_patient(graph, thread_id)
    if stored is not None and stored.checkpoint not in (None, setup.identity):
        print(f"Warning: stored memory for {subject} came from a different checkpoint ({stored.checkpoint}).")
    print(
        f"Patient {subject} | target {clip_id} | conditioning {setup.conditioning} | "
        f"memory {parse_memory_spec(memory_spec)[0]} (thread {thread_id}) | "
        f"prior exercises in memory: {0 if stored is None else len(stored.exercises)}"
    )

    target = source.find(clip_id, subject)
    prompt = source.prompt(target)
    print(f"Model prompt ({setup.conditioning} conditioning):\n  " + prompt.replace("\n", "\n  "))
    if replay:
        ordered = source.exercises(subject, args.order)
        earlier = ordered[: [e.clip_id for e in ordered].index(clip_id)]
        seen = set() if stored is None else stored.seen_clips()
        earlier = [e for e in earlier if e.clip_id not in seen]
        if max_history is not None:
            earlier = earlier[-max_history:]
        print(f"Replaying {len(earlier)} earlier exercises through the loop (recorded, not paced) ...")
        history_session = RehabSession(graph, services, thread_id, seed)
        history = tqdm(earlier, desc="history", unit="exercise", dynamic_ncols=True)
        for exercise in history:
            history.set_postfix_str(exercise.clip_id)
            exercise_report = history_session.replay(exercise, progress=True)
            record = exercise_report["record"]
            tqdm.write(
                f"  #{record['exercise_index'] + 1:>3} {exercise.clip_id:28s} "
                f"decision={exercise_report['decision']:8s} "
                f"released={record['released_windows']}/{record['n_windows']} "
                f"UE-FMA estimate={record['ue_fma_estimate_after'] or float('nan'):.1f}"
            )
    stored = load_patient(graph, thread_id)
    preview = stored is not None and clip_id in stored.seen_clips() and not args.record_repeat
    if preview:
        # Re-counting the same clip would inflate the patient's evidence (UE-FMA CI,
        # primitive counts), so a repeat runs on a scratch thread seeded with the stored
        # memory and is neither saved nor logged to the persistent store.
        print(
            f"Note: {clip_id} is already in {subject}'s history; running it as a preview with that "
            "history (memory not updated). Use --record-repeat to append it instead."
        )
        services.log = WindowLog(log.namespace)
        preview_thread = f"{thread_id}:preview:{uuid.uuid4().hex[:8]}"
        session = RehabSession(graph, services, preview_thread, seed, patient=stored.model_dump())
    else:
        if stored is not None and clip_id in stored.seen_clips():
            print(f"Note: {clip_id} is already in {subject}'s history; appending it again (--record-repeat).")
        session = RehabSession(graph, services, thread_id, seed)
    fps = args.fps or STREAM_FPS
    target_start = len(services.log.records)
    exercise_report, stream_rows = stream_exercise(
        session, services, source, target, fps=fps, paced=not args.fast, print_mode=args.print_mode
    )
    if preview:
        delete_thread(checkpointer, preview_thread)
    print("\n" + exercise_report["text"])
    target_records = list(services.log.records[target_start:])

    output = args.output_dir / (scenario.name if scenario else clip_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.txt").write_text(exercise_report["text"] + "\n", encoding="utf-8")
    (output / "report_details.txt").write_text(exercise_report["details"] + "\n", encoding="utf-8")
    _write_csv(output / "recommendations.csv", stream_rows)
    (output / "model_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    with (output / "window_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in target_records:
            handle.write(json.dumps(record) + "\n")
    patient = load_patient(graph, thread_id)
    (output / "patient_memory.json").write_text(patient.model_dump_json(indent=2), encoding="utf-8")

    # History for the trajectory: every stored window of this patient (this run's target
    # replaces any older run of the same clip) and the matching exercise records.
    history_records = [r for r in log.stored([subject]) if not (preview and r["clip_id"] == clip_id)]
    exercise_records = [record.model_dump() for record in patient.exercises]
    if preview:
        history_records += target_records
        exercise_records.append(exercise_report["record"])
    figures = _figures(source, target_records, history_records, exercise_records, subject, clip_id, output, config)
    if scenario and scenario.compare_without_memory:
        _compare_without_memory(graph, services, source, target, seed, target_records, output)
    print(
        f"\nWrote {output}/report.txt, report_details.txt, recommendations.csv, window_decisions.jsonl, patient_memory.json"
        + "".join(f", {f.name}" for f in figures)
    )


def stream_exercise(session, services, source, exercise, *, fps: float, paced: bool, print_mode: str):
    """Feed the clip frame by frame and print each recommendation as it is made."""
    stride = train_config.WINDOW_STRIDE
    span = (train_config.CONTEXT_WINDOW_SIZE - 1) * train_config.FRAME_SKIP + 1
    budget_ms = 1000.0 * stride / fps
    services.perception.warm_up()  # load the model before the clock starts
    session.begin(exercise, streaming=True)
    print(
        f"\nStreaming {exercise.clip_id} at {fps:g} fps ({'real time' if paced else 'fast, unpaced'}): "
        f"buffering {span} frames ({span / fps:.1f} s), then a recommendation every {stride} frames "
        f"({stride / fps:.2f} s, the latency budget)."
    )
    rows, last_printed = [], None
    buffering = tqdm(total=span, desc="buffering", unit="frame", leave=False, dynamic_ncols=True)
    clock = time.perf_counter()
    for frame_index, frame in enumerate(source.frames(exercise)):
        if paced:
            delay = clock + frame_index / fps - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        decision = session.push(frame)
        if buffering is not None:
            buffering.update(1)
            if decision is not None or frame_index + 1 >= span:
                buffering.close()
                buffering = None
        if decision is None:
            continue
        lag = max(0.0, time.perf_counter() - (clock + (frame_index + 1) / fps)) if paced else None
        row = {
            "stream_time_s": round(frame_index / fps, 3),
            "frame": frame_index,
            "window_index": decision["window_index"],
            "current": decision["current_primitive"],
            "current_confidence": round(decision["confidence"][0], 4),
            "future": decision["next_primitive"],
            "future_confidence": round(decision["next_confidence"], 4),
            "status": decision["status"],
            "released": decision["released"],
            "feedback": decision["feedback"],
            "latency_ms": round(decision["latency_ms"], 1),
            "lag_s": None if lag is None else round(lag, 3),
        }
        rows.append(row)
        key = (row["current"], row["future"], row["status"])
        if print_mode == "every" or key != last_printed:
            cue = (
                f"current {row['current']:<10s} {row['current_confidence']:.2f} | future {row['future']:<10s} {row['future_confidence']:.2f}"
                if row["released"]
                else row["feedback"]
            )
            print(f"  t={row['stream_time_s']:7.2f}s frame {frame_index:5d} | {cue} | {row['status'].upper():8s} | {row['latency_ms']:6.0f} ms")
            last_printed = key
    if buffering is not None:
        buffering.close()
    exercise_report = session.end()
    if rows:
        latency = np.asarray([r["latency_ms"] for r in rows])
        summary = (
            f"{len(rows)} recommendations | step latency mean {latency.mean():.0f} ms, p95 {np.percentile(latency, 95):.0f} ms | "
            f"within the {budget_ms:.0f} ms budget: {np.mean(latency <= budget_ms):.0%}"
        )
        if paced:
            summary += f" | max lag behind real time {max(r['lag_s'] for r in rows):.2f} s"
        print(summary)
    else:
        print(f"No recommendation: the clip is shorter than the {span}-frame window.")
    return exercise_report, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figures(source, target_records, history_records, exercise_records, subject, clip_id, output: Path, config: MASConfig) -> list[Path]:
    """Decision timeline of this exercise and the patient's trajectory across exercises.

    Ground truth is joined here, for the figures only; it never enters agent state.
    """
    import pandas as pd

    from multi_agents import metrics, plots

    records = pd.DataFrame(target_records)
    primitives = metrics.primitive_frame(records, source, "mas", "demo")
    figures = [plots.decision_timeline(records, primitives, clip_id, output / "decision_timeline.png", subject_id=subject)]
    history = pd.DataFrame(history_records)
    if len(history) and history["clip_id"].nunique() > 1:
        history = history.drop_duplicates(["clip_id", "window_index"], keep="last")
        history_primitives = metrics.primitive_frame(history, source, "mas", "demo")
        mas = metrics.exercise_frame(history, history_primitives, source, "mas", {subject: exercise_records}, config)
        vlm = mas.assign(mode="vlm", accuracy=mas["model_accuracy"], ue_estimate=mas["ue_model_exercise"])
        figures.append(
            plots.patient_trajectory({"vlm": vlm, "mas": mas}, subject, output / "patient_trajectory.png")
        )
    return figures


def _compare_without_memory(graph, services, source, target, seed, with_memory, output: Path) -> None:
    import pandas as pd

    from multi_agents import metrics

    services.log.records.clear()
    RehabSession(graph, services, f"ablation:{uuid.uuid4().hex}", seed).replay(target)
    rows = []
    for label, records in (("with memory", with_memory), ("without memory", list(services.log.records))):
        frame = metrics.primitive_frame(pd.DataFrame(records), source, "mas", "demo")
        rows.append(
            {
                "run": label,
                "accuracy": float((frame["actual_label_id"] == frame["predicted_label_id"]).mean()),
                "vlm_accuracy": float((frame["actual_label_id"] == frame["model_label_id"]).mean()),
                "released_windows": int(pd.DataFrame(records)["released"].sum()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "memory_ablation.csv", index=False)
    print("\nMemory ablation on the target exercise (ground truth used only for this table):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
