"""Real-time demonstration: stream one exercise through the multi-agent system.

    python -m multi_agents.demo --sample-id s04_drinking3_1 --replay-history
    python -m multi_agents.demo --sample-id s04_drinking3_1 --fast

The target clip is fed frame by frame at the feature frame rate (15 fps by default,
`mas.stream_fps` in config.yaml), as a live sensor would deliver it. After 116
frames of warm-up, every 5th frame completes a window and the agents return a
recommendation: the current primitive and the future primitive to follow (4 frames, 0.27 s
ahead). `--replay-history` first runs the patient's earlier exercises from their
recorded model outputs, so the live exercise is decided with that history.

Patient history is kept in a sqlite checkpointer by default, so running the demo again
continues the same patient's record (`--memory inmemory` for a fresh, isolated run).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import uuid
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from multi_agents.data import open_source  # noqa: E402
from multi_agents.demo_outputs import export_demo  # noqa: E402
from multi_agents.graph import build_graph  # noqa: E402
from multi_agents.memory import WindowLog, delete_thread, load_patient, make_checkpointer, parse_memory_spec  # noqa: E402
from multi_agents.priors import resolve_priors  # noqa: E402
from multi_agents.session import RehabSession  # noqa: E402
from multi_agents.settings import (  # noqa: E402
    DEMO_OUTPUT_DIR,
    DEFAULT_SQLITE_PATH,
    LOG_DIR,
    RAW_DATA_DIR,
    STREAM_FPS,
    MASConfig,
)
from multi_agents.system import (  # noqa: E402
    build_services,
    log_namespace,
    make_backend,
    patient_seed,
    prepare_model,
)
from multi_agents.train_bridge import start_log, train_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-id", required=True, help="Clip id of a test patient, e.g. s04_drinking3_1")
    parser.add_argument("--replay-history", action="store_true", help="Run the patient's earlier exercises first")
    parser.add_argument("--history-clips", type=int, default=None,
                        help="Replay at most this many immediately preceding clips; 0 replays none (omitted: all with --replay-history)")
    parser.add_argument("--fast", action="store_true", help="Stream without real-time pacing (same outputs)")
    parser.add_argument("--checkpoint", default=None, help="Default: paths.checkpoint in config.yaml")
    parser.add_argument("--mas-config", type=Path, default=None, help="tuned_config.json from multi_agents.tune")
    parser.add_argument("--memory", default=f"sqlite:{DEFAULT_SQLITE_PATH}", help="'inmemory' or 'sqlite:<path>'")
    parser.add_argument("--output-dir", type=Path, default=DEMO_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.history_clips is not None and args.history_clips < 0:
        parser.error("--history-clips cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    start_log(LOG_DIR, "demo")
    clip_id = args.sample_id
    subject = clip_id.split("_", 1)[0]
    config = MASConfig.from_json(args.mas_config)
    setup = prepare_model(args.checkpoint, args.base_model, not args.no_download)
    source = open_source(
        split="test", raw_dir=args.raw_dir, conditioning=setup.conditioning, normalizer=setup.normalizer, subjects=[subject]
    )
    priors = resolve_priors(args.raw_dir)

    checkpointer, _ = make_checkpointer(args.memory)
    backend = make_backend(source, setup, args.device, args.batch_size)
    services = build_services(
        config=config, priors=priors, backend=backend, log=WindowLog(log_namespace(config, setup)), mode="mas"
    )
    graph = build_graph(services, checkpointer)
    thread_id = subject
    seed = patient_seed(source, subject, config, setup)

    stored = load_patient(graph, thread_id)
    if stored is not None and stored.checkpoint not in (None, setup.identity):
        print(f"Warning: stored memory for {subject} came from a different checkpoint ({stored.checkpoint}).")
    print(
        f"Patient {subject} | target {clip_id} | conditioning {setup.conditioning} | "
        f"memory {parse_memory_spec(args.memory)[0]} (thread {thread_id}) | "
        f"prior exercises in memory: {0 if stored is None else len(stored.exercises)}"
    )

    target = source.find(clip_id, subject)
    print(f"Model prompt ({setup.conditioning} conditioning):\n  " + source.prompt(target).replace("\n", "\n  "))
    if args.replay_history or args.history_clips is not None:
        ordered = source.exercises(subject)
        earlier = ordered[: [e.clip_id for e in ordered].index(clip_id)]
        if args.history_clips is not None:
            earlier = earlier[-args.history_clips:] if args.history_clips else []
        seen = set() if stored is None else stored.seen_clips()
        earlier = [e for e in earlier if e.clip_id not in seen]
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
    preview = stored is not None and clip_id in stored.seen_clips()
    if preview:
        # Re-counting the same clip would inflate the patient's evidence (UE-FMA CI,
        # primitive counts), so a repeat runs on a scratch thread seeded with the stored
        # memory and is not saved.
        print(f"Note: {clip_id} is already in {subject}'s history; running it as a preview with that history (memory not updated).")
        preview_thread = f"{thread_id}:preview:{uuid.uuid4().hex[:8]}"
        session = RehabSession(graph, services, preview_thread, seed, patient=stored.model_dump())
    else:
        session = RehabSession(graph, services, thread_id, seed)
    first_record = len(services.log.records)
    exercise_report, stream_rows = stream_exercise(session, services, source, target, fps=STREAM_FPS, paced=not args.fast)
    result_patient = load_patient(graph, session.thread_id)
    if preview:
        delete_thread(checkpointer, preview_thread)
    print("\n" + exercise_report["text"])

    output = args.output_dir / clip_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.txt").write_text(exercise_report["text"] + "\n", encoding="utf-8")
    (output / "report_details.txt").write_text(exercise_report["details"] + "\n", encoding="utf-8")
    if stream_rows:
        with (output / "recommendations.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stream_rows[0]))
            writer.writeheader()
            writer.writerows(stream_rows)
    elif (output / "recommendations.csv").exists():
        (output / "recommendations.csv").unlink()
    export_demo(output, source, services.log.records[first_record:], result_patient,
                load_patient(graph, thread_id), config, source.prompt(target), preview)
    print(f"\nWrote reports, recommendations, agent traces, memory and diagnostic plots to {output}")


def stream_exercise(session, services, source, exercise, *, fps: float, paced: bool):
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
    rows = []
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
        }
        rows.append(row)
        cue = (
            f"current {row['current']:<10s} {row['current_confidence']:.2f} | future {row['future']:<10s} {row['future_confidence']:.2f}"
            if row["released"]
            else row["feedback"]
        )
        print(f"  t={row['stream_time_s']:7.2f}s frame {frame_index:5d} | {cue} | {row['status'].upper():8s} | {row['latency_ms']:6.0f} ms")
    if buffering is not None:
        buffering.close()
    exercise_report = session.end()
    if rows:
        latency = np.asarray([r["latency_ms"] for r in rows])
        print(
            f"{len(rows)} recommendations | step latency mean {latency.mean():.0f} ms, p95 {np.percentile(latency, 95):.0f} ms | "
            f"within the {budget_ms:.0f} ms budget: {np.mean(latency <= budget_ms):.0%}"
        )
    else:
        print(f"No recommendation: the clip is shorter than the {span}-frame window.")
    return exercise_report, rows


if __name__ == "__main__":
    main()
