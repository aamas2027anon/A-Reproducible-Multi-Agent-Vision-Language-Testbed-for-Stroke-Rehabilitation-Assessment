"""Real-time session: a live stream of frames in, a recommendation every window out.

    session = RehabSession(graph, services, thread_id, patient_seed)
    session.begin(exercise, streaming=True)
    for frame in camera:                      # 15 fps
        recommendation = session.push(frame)  # None until a window completes
        if recommendation:
            show(recommendation["feedback"])  # current X -> future Y
    report = session.end()

Every 5th frame completes a window (after 116 frames of warm-up); `push` then runs
the full agent cycle for that window and returns the decision. `replay` drives the
same cycle from a recorded clip's cached model outputs, which is how history is
replayed and how the evaluation runs.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from multi_agents.data import Exercise
from multi_agents.graph import Services
from multi_agents.memory import thread_config
from multi_agents.state import ExerciseContext

# One window event runs 5 nodes (observe, propose, safety_check, coordinate, deliver).
RECURSION_LIMIT = 64


class RehabSession:
    def __init__(
        self,
        graph,
        services: Services,
        thread_id: str,
        patient_seed: dict[str, Any],
        patient: dict[str, Any] | None = None,
    ) -> None:
        """`patient` seeds a fresh thread from existing memory (e.g. a preview run)."""
        self.graph = graph
        self.services = services
        self.thread_id = thread_id
        self.patient_seed = patient_seed
        self._patient = patient
        self.state: dict[str, Any] = {}

    def _invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        config = thread_config(self.thread_id, RECURSION_LIMIT)
        try:
            self.state = self.graph.invoke(inputs, config, durability="exit")
        except TypeError as error:  # langgraph < 0.6 has no durability argument
            if "durability" not in str(error):
                raise
            self.state = self.graph.invoke(inputs, config)
        return self.state

    def begin(self, exercise: Exercise, streaming: bool) -> dict[str, Any]:
        """Open an exercise; returns the recorded window count (None when streaming)."""
        context = ExerciseContext(
            key=exercise.key,
            subject_id=exercise.subject_id,
            clip_id=exercise.clip_id,
            task_name=exercise.task_name,
            streaming=streaming,
        )
        inputs: dict[str, Any] = {"event": "start", "exercise": context.model_dump(), "patient_seed": self.patient_seed}
        if self._patient is not None:
            inputs["patient"], self._patient = self._patient, None
        self._invoke(inputs)
        return {"streaming": streaming, "n_windows": None if streaming else self._recorded_windows()}

    def push(self, frame: np.ndarray) -> dict[str, Any] | None:
        """Feed one live frame; returns the recommendation when a window completes."""
        if not self.services.perception.push_frame(frame):
            return None
        return self.step()

    def step(self) -> dict[str, Any]:
        """Run the agent cycle on the newest window."""
        started = time.perf_counter()
        state = self._invoke({"event": "window"})
        return {**state["decision"], "latency_ms": 1000.0 * (time.perf_counter() - started)}

    def end(self) -> dict[str, Any]:
        """Close the exercise: UE-FMA update, exercise review, report, memory."""
        return self._invoke({"event": "end"})["exercise_report"]

    def replay(self, exercise: Exercise, progress: bool = False) -> dict[str, Any]:
        """Run a recorded exercise window by window from its cached model outputs."""
        info = self.begin(exercise, streaming=False)
        windows = range(info["n_windows"])
        if progress:
            windows = tqdm(windows, desc=f"agents {exercise.clip_id}", unit="window", leave=False, dynamic_ncols=True)
        for _ in windows:
            self.step()
        return self.end()

    def _recorded_windows(self) -> int:
        return int(self.services.perception.backend.run(self.state["exercise"]["key"]).n_windows)
