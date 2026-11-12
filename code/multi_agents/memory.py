"""Patient memory: LangGraph checkpointers and the window log.

Patient, primitive and agent history live in the graph state of the patient's
thread; the checkpointer persists it after every exercise.

* `inmemory` -> `InMemorySaver`: isolated, reproducible runs (evaluation).
* `sqlite:<path>` -> `SqliteSaver`: the demo's history, which persists across runs.

Window-level decisions are too fine-grained for graph state, so they go to a
`WindowLog` in memory, which the evaluation scores after a run.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from multi_agents.state import PatientMemory


def parse_memory_spec(spec: str) -> tuple[str, Path | None]:
    if spec == "inmemory":
        return "inmemory", None
    if spec.startswith("sqlite:"):
        return "sqlite", Path(spec.removeprefix("sqlite:")).expanduser().resolve()
    raise ValueError(f"--memory must be 'inmemory' or 'sqlite:<path>', got {spec!r}")


def make_checkpointer(spec: str):
    """Return (checkpointer, sqlite_path or None)."""
    kind, path = parse_memory_spec(spec)
    if kind == "inmemory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), None
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(connect_sqlite(path)), path


def connect_sqlite(path: Path) -> sqlite3.Connection:
    """A connection the checkpointer can share with its background writer thread.

    WAL mode lets readers and one writer proceed together, and the busy timeout makes a
    writer wait for the lock instead of failing with "database is locked".
    """
    connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def thread_config(thread_id: str, recursion_limit: int = 100) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def load_patient(graph, thread_id: str) -> PatientMemory | None:
    snapshot = graph.get_state(thread_config(thread_id))
    values = snapshot.values if snapshot is not None else None
    patient = values.get("patient") if isinstance(values, dict) else None
    return None if patient is None else PatientMemory.model_validate(patient)


def delete_thread(checkpointer, thread_id: str) -> None:
    if hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(thread_id)


class WindowLog:
    """Window-level decision records of one run."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
