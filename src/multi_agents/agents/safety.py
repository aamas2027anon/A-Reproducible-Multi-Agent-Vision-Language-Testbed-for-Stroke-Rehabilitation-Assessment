"""Safety agent: deterministic rules only.

Every window and exercise verdict comes from the rules below. The escalation threshold
depends on the patient's impairment band, which comes from the MAS's own UE-FMA
estimate, never from the ground truth.

Window rules, in order:
  R1 non-finite input                         -> block
  R2 current-frame confidence b_0(plan_0) < tau_escalate[band] -> escalate (withhold the cue)
  C1 at each frame h, a cue that is not the a maximizing
     b_h(a) - rest_cost * b_h(rest) * [a is a movement]   -> revise to that a
Exercise rules:
  E1 recent escalation rate above the limit   -> clinician review
  E2 UE-FMA drift flagged by the evaluation agent -> clinician review

`safety_enabled=False` is the no-safety ablation: every usable window is approved and
no exercise rule fires (R1 still blocks unusable input).
"""

from __future__ import annotations

import numpy as np

from multi_agents.settings import ACTIVE_IDS, ACTIVE_PRIMITIVES, PRIMITIVES, REST_ID, MASConfig
from multi_agents.state import PerceptionReport, Proposal, SafetyVerdict
from multi_agents.tools import ToolBox, make_tool


def plain_reason(reason: str) -> str:
    """A clinician-readable version of a safety rule's reason."""
    if reason.startswith("R1"):
        return "the sensor data was unusable"
    if reason.startswith("E1"):
        return "prompts were often held back in recent exercises"
    if reason.startswith("E2"):
        return "this exercise's arm-function estimate differs from the patient's usual level"
    return reason


class SafetyAgent:
    name = "safety"

    def __init__(self, config: MASConfig) -> None:
        self.config = config
        self.tools = ToolBox(
            self.name,
            [
                make_tool(self._get_patient_constraints, "get_patient_constraints", "Safety thresholds for an impairment band."),
                make_tool(self._check_cue, "check_cue", "Rule hits for one primitive cue at one frame."),
                make_tool(self._get_safety_history, "get_safety_history", "The patient's recent escalation history."),
            ],
        )

    # ------------------------------------------------------------------ tools

    def _get_patient_constraints(self, band: str) -> dict:
        return {"band": band, "escalate_confidence": self.config.escalate_confidence[band]}

    def _cost_optimal(self, posterior: list[float]) -> str:
        """The cue maximizing P(a) - rest_cost * P(rest) * [a is a movement], over all classes."""
        row = np.asarray(posterior, dtype=float)
        utility = row - self.config.rest_cost * row[REST_ID] * np.isin(np.arange(len(row)), ACTIVE_IDS)
        return PRIMITIVES[int(np.argmax(utility))]

    def _check_cue(self, cue: str, posterior: list[float]) -> list[str]:
        best = self._cost_optimal(posterior)
        if best == cue:
            return []
        return [f"C1 cue {cue} not cost-optimal (P(rest)={posterior[REST_ID]:.2f}); use {best}"]

    def _get_safety_history(self, escalation_rates: list[float], current_rate: float) -> dict:
        window = self.config.escalation_history
        recent = [*escalation_rates[-(window - 1):], current_rate] if window > 1 else [current_rate]
        return {"recent_escalation_rate": float(np.mean(recent)), "exercises_considered": len(recent)}

    # ------------------------------------------------------------------ window

    def cue_permitted(self, cue: str, posterior: list[float]) -> bool:
        if not self.config.safety_enabled:
            return True
        return not self.tools("check_cue", cue=cue, posterior=posterior)

    def check_window(self, *, report: PerceptionReport, plan: Proposal, band: str) -> SafetyVerdict:
        if not report.integrity_ok:
            return SafetyVerdict(verdict="block", rule_hits=["R1 non-finite input features"])
        if not self.config.safety_enabled:
            return SafetyVerdict(verdict="approve")

        constraints = self.tools("get_patient_constraints", band=band)
        posteriors = plan.details["posteriors"]
        action = list(plan.action)
        current_confidence = posteriors[0][PRIMITIVES.index(action[0])]
        if current_confidence < constraints["escalate_confidence"]:
            return SafetyVerdict(
                verdict="escalate",
                rule_hits=[f"R2 current-frame confidence {current_confidence:.2f} < {constraints['escalate_confidence']:.2f}"],
            )

        hits, revised = [], []
        for horizon, cue in enumerate(action):
            cue_hits = self.tools("check_cue", cue=cue, posterior=posteriors[horizon])
            hits += [f"h{horizon}: {hit}" for hit in cue_hits]
            revised.append(self._cost_optimal(posteriors[horizon]) if cue_hits else cue)

        changed = revised != action
        return SafetyVerdict(
            verdict="revise" if changed else "approve",
            rule_hits=hits,
            revised_action=revised if changed else None,
        )

    @staticmethod
    def proposal(verdict: SafetyVerdict, plan: Proposal) -> Proposal:
        posteriors = plan.details["posteriors"]
        removed = 0.0
        if verdict.revised_action:
            for horizon, (before, after) in enumerate(zip(plan.action, verdict.revised_action)):
                if before in ACTIVE_PRIMITIVES and after not in ACTIVE_PRIMITIVES:
                    removed += posteriors[horizon][REST_ID]
        return Proposal(
            agent="safety",
            action=verdict.verdict,
            confidence=1.0 if not verdict.rule_hits else 0.5,
            expected_benefit=removed / len(plan.action),
            constraints=verdict.rule_hits,
            rationale="Rule verdict",
            source="rule",
            details={"revised_action": verdict.revised_action},
        )

    # ---------------------------------------------------------------- exercise

    def review_exercise(self, *, patient: dict, summary: dict) -> dict:
        reasons = []
        if summary.get("blocked_reason"):
            reasons.append(f"R1 {summary['blocked_reason']}")
        if not self.config.safety_enabled:
            return {"reasons": reasons, "clinician_flag": bool(reasons)}
        history = self.tools(
            "get_safety_history",
            escalation_rates=patient["agents"]["escalation_rates"],
            current_rate=summary["escalation_rate"],
        )
        if history["recent_escalation_rate"] > self.config.escalation_rate_limit:
            reasons.append(
                f"E1 escalation rate {history['recent_escalation_rate']:.2f} over the last "
                f"{history['exercises_considered']} exercises"
            )
        if summary.get("uefma_drift"):
            reasons.append("E2 UE-FMA of this exercise drifted from the patient estimate")
        return {"reasons": reasons, "clinician_flag": bool(reasons)}
