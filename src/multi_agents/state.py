"""Agent state for the closed-loop StrokeRehab multi-agent system.

Two lifetimes live in one LangGraph state:

* `patient` persists across exercises on the patient's checkpointer thread. It is
  the patient, primitive and agent history other agents read to decide.
* Everything else describes the exercise in progress and is reset by the
  coordinator when an exercise starts.

Graph channels hold plain JSON values (the Pydantic models below are dumped into
them), so every checkpointer backend can serialize the state. Ground truth - primitive
labels, UE_FMA_score and the metadata `impairment_level`, which is a bucket of the
UE-FMA target - is never part of the state; `assert_no_ground_truth` enforces it.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from multi_agents.settings import NUM_PRIMITIVES

Verdict = Literal["approve", "revise", "escalate", "block"]
DecisionStatus = Literal["approve", "revise", "escalate", "block"]
UEFMAUpdate = Literal["accept", "down-weight", "hold"]


class PatientProfile(BaseModel):
    """Static, non-leaky patient facts (the fields the text/prefix models use)."""

    subject_id: str
    paretic_side: str | None = None
    gender: str | None = None
    age: float | None = None
    race: str | None = None
    time_since_stroke: float | None = None


class PerceptionReport(BaseModel):
    """The perception agent's output for one window.

    `probs[h]` is the full class distribution for frame target+h: h=0 is the last
    context frame, h=1..4 are the four future frames.
    """

    window_index: int
    target_frame_index: int
    probs: list[dict[str, float]]
    predicted_labels: list[str]
    confidence: list[float]
    ue_fma: float
    integrity_ok: bool = True


class Proposal(BaseModel):
    """What every agent returns to the coordinator."""

    agent: str
    action: list[str] | str
    confidence: float
    expected_benefit: float
    constraints: list[str] = Field(default_factory=list)
    rationale: str = ""
    source: Literal["model", "rule"] = "rule"
    details: dict[str, Any] = Field(default_factory=dict)


class SafetyVerdict(BaseModel):
    verdict: Verdict
    rule_hits: list[str] = Field(default_factory=list)
    revised_action: list[str] | None = None


class WindowDecision(BaseModel):
    """One real-time recommendation: the cue for frames t..t+4."""

    window_index: int
    target_frame_index: int
    final_action: list[str]
    confidence: list[float]
    current_primitive: str = ""  # frame t
    next_primitive: str = ""  # frame t+4: the next primitive to follow
    next_confidence: float = 0.0
    status: DecisionStatus
    released: bool
    selected_agent: str
    disagreement: bool = False
    coordination_failure: str | None = None
    feedback: str = ""


class PrimitiveMemory(BaseModel):
    """Patient-specific primitive statistics, learned from the MAS's own outputs."""

    transition_counts: list[list[float]] = Field(
        default_factory=lambda: [[0.0] * NUM_PRIMITIVES for _ in range(NUM_PRIMITIVES)]
    )
    last_released: str | None = None


class UEFMAMemory(BaseModel):
    exercise_estimates: list[float] = Field(default_factory=list)
    exercise_weights: list[float] = Field(default_factory=list)
    mean: float | None = None  # raw scale (pooling)
    mean_cal: float | None = None  # calibrated: clip(ue_calib_a + ue_calib_b * mean, 0, 66)
    variance: float | None = None  # raw scale
    band: str | None = None  # band of mean_cal
    drift_exercises: list[int] = Field(default_factory=list)


class AgentMemory(BaseModel):
    proposals: dict[str, int] = Field(default_factory=dict)
    selected: dict[str, int] = Field(default_factory=dict)
    overridden: dict[str, int] = Field(default_factory=dict)
    safety_verdicts: dict[str, int] = Field(default_factory=dict)
    escalation_rates: list[float] = Field(default_factory=list)


class ExerciseRecord(BaseModel):
    """Appended to the patient's history after every exercise."""

    exercise_index: int
    clip_id: str
    task_name: str
    n_windows: int
    status_counts: dict[str, int]
    released_windows: int
    mean_confidence: float | None
    # UE-FMA fields are calibrated (MASConfig.calibrate_uefma); pooling stays on the raw scale.
    ue_fma_exercise: float | None
    ue_fma_estimate_after: float | None  # mu_cal
    ue_fma_ci95: float | None  # 1.96 * sigma_cal
    ue_fma_estimate_after_raw: float | None = None  # mu, uncalibrated (comparison with raw numbers)
    ue_fma_update: UEFMAUpdate
    predicted_counts: dict[str, int]
    disagreements: int
    revisions: int
    coordination_failures: int
    latency_ms: dict[str, float]
    clinician_flag: bool
    clinician_reasons: list[str] = Field(default_factory=list)


class PatientMemory(BaseModel):
    profile: PatientProfile
    primitive: PrimitiveMemory = Field(default_factory=PrimitiveMemory)
    ue_fma: UEFMAMemory = Field(default_factory=UEFMAMemory)
    agents: AgentMemory = Field(default_factory=AgentMemory)
    exercises: list[ExerciseRecord] = Field(default_factory=list)
    config_hash: str | None = None
    checkpoint: str | None = None

    def seen_clips(self) -> set[str]:
        return {record.clip_id for record in self.exercises}


class ExerciseContext(BaseModel):
    key: str
    subject_id: str
    clip_id: str
    task_name: str
    exercise_index: int = 0
    n_windows: int = 0  # windows observed so far (unknown up front when streaming)
    streaming: bool = False


class RehabState(TypedDict, total=False):
    # Which path this invocation runs: "start" | "window" | "end".
    event: str
    # Persists across exercises (patient thread).
    patient: dict[str, Any]
    # Supplied by the caller for each exercise.
    exercise: dict[str, Any]
    patient_seed: dict[str, Any]
    # Reset at the start of every exercise.
    cursor: int
    report: dict[str, Any] | None
    belief: list[float] | None
    proposals: dict[str, dict[str, Any]]
    verdict: dict[str, Any] | None
    decision: dict[str, Any] | None
    stats: dict[str, Any]
    blocked_reason: str | None
    exercise_report: dict[str, Any] | None


FORBIDDEN_STATE_KEYS = frozenset(
    {
        "gesture_seq",
        "gesture_labels",
        "actual_primitive",
        "actual_label_id",
        "actual_ue_fma",
        "ue_fma_score",
        "UE_FMA_score",
        "impairment_level",
    }
)


def assert_no_ground_truth(value: Any, path: str = "state") -> None:
    """Fail loudly if a ground-truth field ever reaches agent state."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_STATE_KEYS:
                raise AssertionError(f"Ground-truth field {path}.{key} reached agent state")
            assert_no_ground_truth(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value[:3]):
            assert_no_ground_truth(item, f"{path}[{index}]")
