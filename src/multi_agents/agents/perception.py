"""Perception agent: the only agent with access to the trained movement model.

An exercise is observed in one of two ways, behind the same per-window `observe`:

* streaming: frames arrive one at a time (`push_frame`); a `WindowAssembler` emits
  the training pipeline's windows as they complete and the model scores each one live;
* replay: the model's batched, cached outputs for a recorded clip are read back window
  by window (history replay and evaluation, where the model has already run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from multi_agents.data import ExerciseSource
from multi_agents.inference import (
    ExerciseInference,
    InferenceCache,
    MovementModelRunner,
    WindowAssembler,
    WindowResult,
    window_integrity,
    window_report,
)
from multi_agents.settings import NUM_PRIMITIVES
from multi_agents.state import PerceptionReport, Proposal
from multi_agents.tools import ToolBox, make_tool


class ModelBackend:
    """Owns the model; scores whole recorded clips (batched, cached) or single windows.

    The runner is built lazily, so an exercise whose every window fails the integrity
    check never loads the model or touches the GPU.
    """

    def __init__(
        self,
        source: ExerciseSource,
        runner_factory: Callable[[], MovementModelRunner],
        cache: InferenceCache | None = None,
    ) -> None:
        self.source = source
        self._runner_factory = runner_factory
        self._runner: MovementModelRunner | None = None
        self.cache = cache
        self._memo: dict[str, ExerciseInference] = {}

    def runner(self) -> MovementModelRunner:
        if self._runner is None:
            self._runner = self._runner_factory()
            if self._runner.conditioning != self.source.dataset.conditioning:
                raise ValueError(
                    f"Checkpoint expects {self._runner.conditioning!r} conditioning but the data "
                    f"source builds {self.source.dataset.conditioning!r} inputs"
                )
        return self._runner

    def run(self, exercise_key: str) -> ExerciseInference:
        """All windows of a recorded clip (batched pass, memoized and disk-cached)."""
        if exercise_key not in self._memo:
            self._memo[exercise_key] = self._compute(exercise_key)
        return self._memo[exercise_key]

    def _compute(self, exercise_key: str) -> ExerciseInference:
        if self.cache is not None and (cached := self.cache.get(exercise_key)) is not None:
            return cached
        subject_id, clip_id = exercise_key.split("/", 1)
        exercise = self.source.find(clip_id, subject_id)
        samples = self.source.window_samples(exercise)
        target_frames = self.source.target_frames(exercise)
        window_ok = window_integrity(samples)
        if not window_ok.any():
            result = ExerciseInference(
                probs=np.full((len(samples), 5, NUM_PRIMITIVES), 1.0 / NUM_PRIMITIVES, np.float32),
                ue_fma=np.full(len(samples), np.nan, np.float32),
                target_frames=target_frames,
                window_ok=window_ok,
                latency_ms=0.0,
            )
        else:
            result = self.runner().predict(samples, target_frames, progress=f"model  {clip_id}")
        if self.cache is not None:
            self.cache.put(exercise_key, result)
        return result

    def score_window(self, sample: dict, target_frame: int) -> WindowResult:
        """One live window (streaming)."""
        if not window_integrity([sample])[0]:
            return WindowResult(
                probs=np.full((5, NUM_PRIMITIVES), 1.0 / NUM_PRIMITIVES, np.float32),
                ue_fma=float("nan"),
                target_frame=target_frame,
                ok=False,
                model_ms=0.0,
            )
        return self.runner().predict([sample], np.asarray([target_frame])).window(0)


@dataclass
class _OpenExercise:
    key: str
    streaming: bool
    assembler: WindowAssembler | None = None
    recorded: ExerciseInference | None = None
    next_window: int = 0


class PerceptionAgent:
    name = "perception"

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self._open: _OpenExercise | None = None
        self.last_model_ms = 0.0  # model compute attributed to the latest window
        self.last_spent_ms = 0.0  # model compute actually spent during the latest observe
        self.tools = ToolBox(
            self.name,
            [
                make_tool(
                    self._open_exercise,
                    "open_exercise",
                    "Start observing an exercise, live (streaming) or from its recorded model outputs.",
                ),
                make_tool(
                    self._run_movement_model,
                    "run_movement_model",
                    "Score the newest complete window with the fine-tuned Qwen movement model.",
                ),
            ],
        )

    # ------------------------------------------------------------------ tools

    def _open_exercise(self, exercise_key: str, streaming: bool) -> dict:
        if streaming:
            subject_id, clip_id = exercise_key.split("/", 1)
            exercise = self.backend.source.find(clip_id, subject_id)
            context = self.backend.source.stream_context(exercise)
            self._open = _OpenExercise(
                exercise_key, True, assembler=WindowAssembler(context["text"], context["patient_features"])
            )
            return {"streaming": True, "n_windows": None}
        recorded = self.backend.run(exercise_key)
        self._open = _OpenExercise(exercise_key, False, recorded=recorded)
        return {"streaming": False, "n_windows": recorded.n_windows}

    def _run_movement_model(self, exercise_key: str) -> dict:
        current = self._require(exercise_key)
        if current.streaming:
            sample, target_frame = current.assembler.take()
            result = self.backend.score_window(sample, target_frame)
            spent = result.model_ms
        else:
            result = current.recorded.window(current.next_window)
            spent = 0.0  # already computed in the batched pass
        index = current.next_window
        current.next_window += 1
        self.last_model_ms, self.last_spent_ms = result.model_ms, spent
        return window_report(index, result).model_dump()

    # ------------------------------------------------------------------ API

    def warm_up(self) -> None:
        """Load the model before a live stream starts, so the first step is not slow."""
        self.backend.runner()

    def open(self, exercise_key: str, streaming: bool) -> dict:
        return self.tools("open_exercise", exercise_key=exercise_key, streaming=streaming)

    def push_frame(self, frame: np.ndarray) -> bool:
        """Buffer one live frame; True when a window is ready to observe."""
        if self._open is None or not self._open.streaming:
            raise RuntimeError("No streaming exercise is open")
        return self._open.assembler.push(frame)

    def observe(self, exercise_key: str) -> PerceptionReport:
        return PerceptionReport.model_validate(self.tools("run_movement_model", exercise_key=exercise_key))

    def close(self, exercise_key: str) -> None:
        if self._open is not None and self._open.key == exercise_key:
            self._open = None

    def _require(self, exercise_key: str) -> _OpenExercise:
        if self._open is None or self._open.key != exercise_key:
            raise RuntimeError(f"Exercise {exercise_key} is not open")
        return self._open

    @staticmethod
    def propose(report: PerceptionReport) -> Proposal:
        confidence = float(np.mean(report.confidence))
        return Proposal(
            agent="perception",
            action=list(report.predicted_labels),
            confidence=confidence,
            expected_benefit=confidence,
            constraints=[] if report.integrity_ok else ["non_finite_input"],
            rationale="Arg-max of the fine-tuned model for frames t..t+4.",
            source="model",
        )
