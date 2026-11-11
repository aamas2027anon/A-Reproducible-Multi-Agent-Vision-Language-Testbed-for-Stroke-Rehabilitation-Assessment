"""Copy everything a run prints to a log file under its output directory."""
from __future__ import annotations

import sys
import time
from pathlib import Path


class _Tee:
    """A text stream that writes to the console stream and to the log file."""

    def __init__(self, stream, log) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(directory: Path, name: str) -> Path:
    """Create `directory` and copy stdout and stderr to `<directory>/<name>_<time>.log`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    print(f"Log: {path}")
    return path
