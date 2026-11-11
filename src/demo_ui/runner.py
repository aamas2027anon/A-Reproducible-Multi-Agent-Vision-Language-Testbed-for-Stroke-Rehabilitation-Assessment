"""Execute the demo in its own process; keep model and agent logic outside the UI."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from multi_agents import settings  # initializes the shared config import path
import config as config

CODE_DIR = Path(__file__).resolve().parents[1]
_RUN_LOCK = threading.Lock()
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def input_defaults() -> tuple[str, str, str]:
    """Resolve one configured checkpoint without loading a model or downloading weights."""
    seed = config.SEEDS[0]
    checkpoint = str(settings.DEFAULT_CHECKPOINT or "")
    notes = []
    if not checkpoint:
        try:
            from multi_agents.checkpoints import text_checkpoints

            checkpoint = str(text_checkpoints(config.TRAINING_RUN_DIR, seeds=(seed,))[seed])
        except (SystemExit, OSError, ValueError, RuntimeError, KeyError) as error:
            notes.append(f"Enter a checkpoint path. Automatic selection: {error}")
    tuned = config.MAS_RUN_DIR / f"seed_{seed}" / "tuned_config.json"
    if settings.DEFAULT_CHECKPOINT:
        paired_settings = Path(settings.DEFAULT_CHECKPOINT).with_name("tuned_config.json")
        # Prefer the settings packaged with an explicitly selected release checkpoint.
        tuned = paired_settings if paired_settings.is_file() else None
    if tuned is None or not tuned.is_file():
        notes.append("No matching tuned settings found; select a JSON file, or leave it blank to use YAML agent settings.")
    if settings.DEFAULT_CHECKPOINT:
        notes.append("Checkpoint comes from paths.checkpoint; confirm the MAS settings match it.")
    return checkpoint, str(tuned) if tuned is not None and tuned.is_file() else "", "\n\n".join(notes)


@dataclass(frozen=True)
class DemoOptions:
    sample_id: str
    checkpoint: Path
    mas_config: Path | None
    history_mode: str
    history_clips: int
    fresh_memory: bool
    persistent_memory: Path | None
    raw_dir: Path
    base_model: Path | None
    device: str
    batch_size: int
    fast: bool
    allow_download: bool
    output_root: Path

    def validate(self) -> None:
        if not self.sample_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.sample_id):
            raise ValueError("Enter a clip ID such as s04_drinking3_1, without directory separators.")
        if not self.checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {self.checkpoint}")
        if self.mas_config is not None and not self.mas_config.is_file():
            raise ValueError(f"MAS settings not found: {self.mas_config}")
        if not self.raw_dir.is_dir():
            raise ValueError(f"Raw data directory not found: {self.raw_dir}")
        if self.history_mode not in ("None", "Last N clips", "All preceding clips"):
            raise ValueError("Choose a history option.")
        if self.history_clips < 0 or self.batch_size <= 0:
            raise ValueError("History count must be nonnegative and batch size must be positive.")
        if not self.fresh_memory:
            if self.persistent_memory is None:
                raise ValueError("Enter a SQLite path for persistent memory.")
            if self.persistent_memory.is_dir():
                raise ValueError("Persistent memory must be a SQLite file path, not a directory.")

    def command(self, run_dir: Path) -> list[str]:
        memory = run_dir / "memory.sqlite" if self.fresh_memory else self.persistent_memory
        command = [
            sys.executable, "-u", "-m", "multi_agents.demo",
            "--sample-id", self.sample_id, "--checkpoint", str(self.checkpoint),
            "--raw-dir", str(self.raw_dir), "--output-dir", str(run_dir),
            "--memory", f"sqlite:{memory}", "--batch-size", str(self.batch_size),
        ]
        if self.mas_config is not None:
            command.extend(("--mas-config", str(self.mas_config)))
        if self.history_mode == "All preceding clips":
            command.append("--replay-history")
        else:
            count = self.history_clips if self.history_mode == "Last N clips" else 0
            command.extend(("--history-clips", str(count)))
        if self.base_model is not None:
            command.extend(("--base-model", str(self.base_model)))
        if self.device and self.device != "auto":
            command.extend(("--device", self.device))
        if self.fast:
            command.append("--fast")
        if not self.allow_download:
            command.append("--no-download")
        return command


@dataclass(frozen=True)
class RunUpdate:
    status: str
    log: str
    run_dir: Path | None = None
    complete: bool = False
    succeeded: bool = False


def _new_run(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("run_%Y%m%d-%H%M%S")
    suffix = 1
    while True:
        directory = root / (name if suffix == 1 else f"{name}_{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            suffix += 1


def run_demo(options: DemoOptions) -> Iterator[RunUpdate]:
    """Stream progress and retain a full log; allow only one active child process."""
    if not _RUN_LOCK.acquire(blocking=False):
        yield RunUpdate("Busy: another demo is running. Try again when it finishes.", "", complete=True)
        return
    process = None
    run_dir = None
    tail: deque[str] = deque(maxlen=250)
    try:
        options.validate()
        run_dir = _new_run(options.output_root)
        command = options.command(run_dir)
        tail.append(f"$ {shlex.join(command)}\n\nFull log: {run_dir / 'execution.log'}\n")
        yield RunUpdate("Running: preparing data and loading the model…", "".join(tail), run_dir)
        with (run_dir / "execution.log").open("w", encoding="utf-8", buffering=1) as log:
            log.write(tail[0])
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command, cwd=CODE_DIR, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            last_update = 0.0
            for line in process.stdout:
                log.write(line)
                tail.append(_ANSI.sub("", line))
                now = time.monotonic()
                if now - last_update >= 0.3:
                    yield RunUpdate("Running: see progress below.", "".join(tail), run_dir)
                    last_update = now
            return_code = process.wait()
        if return_code:
            yield RunUpdate(
                f"Failed: demo exited with code {return_code}. Full log: {run_dir / 'execution.log'}",
                "".join(tail), run_dir, complete=True,
            )
        else:
            yield RunUpdate("Completed.", "".join(tail), run_dir, complete=True, succeeded=True)
    except (OSError, ValueError) as error:
        yield RunUpdate(f"Failed: {error}", "".join(tail), run_dir, complete=True)
    finally:
        # Closing the UI generator must not leave an untracked model process running.
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
        _RUN_LOCK.release()
