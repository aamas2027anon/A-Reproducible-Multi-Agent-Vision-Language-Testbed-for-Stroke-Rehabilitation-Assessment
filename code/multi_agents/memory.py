"""Patient memory: LangGraph checkpointers, the window log and history export.

Patient, primitive and agent history live in the graph state of the patient's
thread; the checkpointer persists it after every exercise.

* `inmemory` -> `InMemorySaver`: isolated, reproducible runs (evaluation default).
* `sqlite:<path>` -> `SqliteSaver`: the demo's history, which persists across runs.

Window-level decisions are too fine-grained for graph state, so they go to a
`WindowLog`. With a sqlite store the log is written to a table in the same file, so
the demo can draw a patient's whole stored history.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

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


def connect_sqlite(path: Path, autocommit: bool = False) -> sqlite3.Connection:
    """A connection that shares the store safely with the other one.

    The checkpointer and the window log write to the same file, and the checkpointer
    writes after every window from a background thread. WAL mode lets readers and one
    writer proceed together, and the busy timeout makes a writer wait for the lock
    instead of failing with "database is locked".
    """
    connection = sqlite3.connect(
        str(path), check_same_thread=False, timeout=30.0, isolation_level=None if autocommit else ""
    )
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
    """Window-level decision records, optionally persisted to a sqlite table."""

    TABLE = "mas_window_log"

    def __init__(self, namespace: str, sqlite_path: Path | None = None) -> None:
        self.namespace = namespace
        self.records: list[dict[str, Any]] = []
        self._connection = None
        if sqlite_path is not None:
            # Autocommit: each record is its own short transaction, so the log never
            # holds the write lock the checkpointer needs between windows.
            self._connection = connect_sqlite(sqlite_path, autocommit=True)
            self._connection.execute(
                f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
                "namespace TEXT, subject_id TEXT, clip_id TEXT, window_index INTEGER, payload TEXT, "
                "PRIMARY KEY (namespace, subject_id, clip_id, window_index))"
            )
            self._connection.commit()

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self._connection is not None:
            self._connection.execute(
                f"INSERT OR REPLACE INTO {self.TABLE} VALUES (?, ?, ?, ?, ?)",
                (
                    self.namespace,
                    record["subject_id"],
                    record["clip_id"],
                    int(record["window_index"]),
                    json.dumps(record),
                ),
            )

    def commit(self) -> None:
        if self._connection is not None:
            self._connection.commit()

    def stored(self, subjects: Iterable[str]) -> list[dict[str, Any]]:
        """Every persisted record of this namespace for the given subjects."""
        if self._connection is None:
            return list(self.records)
        subjects = list(subjects)
        placeholders = ",".join("?" for _ in subjects)
        rows = self._connection.execute(
            f"SELECT payload FROM {self.TABLE} WHERE namespace = ? AND subject_id IN ({placeholders}) "
            "ORDER BY subject_id, clip_id, window_index",
            (self.namespace, *subjects),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]


def export_patient_memory(graph, thread_ids: dict[str, str], path: Path) -> None:
    """Write each patient's final memory and the size of its checkpoint history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for subject_id, thread_id in sorted(thread_ids.items()):
            patient = load_patient(graph, thread_id)
            if patient is None:
                continue
            history = list(graph.get_state_history(thread_config(thread_id)))
            handle.write(
                json.dumps(
                    {
                        "subject_id": subject_id,
                        "thread_id": thread_id,
                        "checkpoints": len(history),
                        "memory": patient.model_dump(),
                    }
                )
                + "\n"
            )
