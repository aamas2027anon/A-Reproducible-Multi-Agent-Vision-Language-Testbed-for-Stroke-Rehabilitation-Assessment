"""Named, reproducible demonstration scenarios.

Each scenario fixes a patient, a target exercise, how much prior interaction is
replayed through the loop first, and an optional input corruption. Scenario runs use
their own memory threads, so they never mix with evaluation or demo history.
"""

from __future__ import annotations

from dataclasses import dataclass

from multi_agents.data import Perturbation


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    subject_id: str
    split: str
    target_clip: str
    replay_history: bool = True
    max_history: int | None = None
    perturbation: Perturbation | None = None
    compare_without_memory: bool = False


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            name="mild_trajectory",
            description="Mild test patient: the plan adapts over 12 prior exercises, then one is reported.",
            subject_id="s26",
            split="test",
            target_clip="s26_drinking3_1",
            max_history=12,
        ),
        Scenario(
            name="moderate_trajectory",
            description="Moderate test patient over 12 prior exercises; tighter safety thresholds.",
            subject_id="s04",
            split="test",
            target_clip="s04_drinking3_1",
            max_history=12,
        ),
        Scenario(
            name="severe_unseen_patient",
            description="Severe patient (clinical UE-FMA 15) from the 'none' split, never used in training. "
            "If the MAS's own calibrated UE-FMA estimate places the patient in the severe band, the "
            "escalation confidence threshold tightens.",
            subject_id="s50",
            split="none",
            target_clip="s50_RTT_left_side5_2",
        ),
        Scenario(
            name="corrupted_features",
            description="Every frame is NaN: each window fails the integrity check (the model is never run "
            "on corrupt input), every cue is blocked, and the exercise is flagged for review.",
            subject_id="s04",
            split="test",
            target_clip="s04_drinking1_1",
            replay_history=False,
            perturbation=Perturbation(kind="nan"),
        ),
        Scenario(
            name="sensor_dropout",
            description="NaN frames in 40-70% of the stream: those windows are blocked mid-stream, and "
            "recommendations resume once 116 clean frames refill the window.",
            subject_id="s04",
            split="test",
            target_clip="s04_drinking2_1",
            replay_history=False,
            perturbation=Perturbation(kind="nan", start_fraction=0.4, end_fraction=0.7),
        ),
        Scenario(
            name="noisy_features",
            description="Heavy feature noise lowers confidence: the safety rules revise and "
            "escalate more windows.",
            subject_id="s26",
            split="test",
            target_clip="s26_feeding2_1",
            replay_history=False,
            perturbation=Perturbation(kind="noise", noise_std=1.5, seed=7),
        ),
        Scenario(
            name="memory_ablation",
            description="Same target exercise with 12 exercises of history vs none.",
            subject_id="s04",
            split="test",
            target_clip="s04_glasses3_1",
            max_history=12,
            compare_without_memory=True,
        ),
    )
}
