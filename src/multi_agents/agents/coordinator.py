"""Coordinator: resolves disagreements under safety limits and reports to the user.

Resolution and the report are deterministic: the report is a template filled by code.

Per window:
  * block / escalate  -> the cue is withheld (the planned cue is logged, not released);
  * revise            -> safety's revision is released (it is the expected-cost optimum
                         under planning's belief);
  * approve           -> where perception and planning disagree, each frame takes the
                         candidate maximizing P(a) - rest_cost * hazard(a), hazard being
                         P(rest) for active cues, restricted to cues safety permits.
Every released cue is re-checked against the safety rules; a violation is logged as a
coordination failure.
"""

from __future__ import annotations

from multi_agents.agents.safety import SafetyAgent
from multi_agents.prompts import CLINICIAN_REPORT_TEMPLATE, SYSTEM_DETAILS_TEMPLATE
from multi_agents.settings import ACTIVE_PRIMITIVES, PRIMITIVES, REST_ID, STREAM_FPS, MASConfig
from multi_agents.state import PerceptionReport, Proposal, SafetyVerdict, WindowDecision
from multi_agents.tools import ToolBox, make_tool


class CoordinatorAgent:
    name = "coordinator"

    def __init__(self, config: MASConfig, safety: SafetyAgent) -> None:
        self.config = config
        self.safety = safety
        self.tools = ToolBox(
            self.name,
            [
                make_tool(self._select_cue, "select_cue", "Pick between disagreeing perception and planning cues."),
                make_tool(
                    self._render_report,
                    "render_report",
                    "Write the clinician report and the system details of an exercise.",
                ),
            ],
        )

    # ------------------------------------------------------------------ tools

    def _select_cue(self, candidates: list[str], posterior: list[float]) -> str:
        def utility(cue: str) -> float:
            hazard = posterior[REST_ID] if cue in ACTIVE_PRIMITIVES else 0.0
            # Arbitration uses safety's tuned cost, so one number drives both.
            return posterior[PRIMITIVES.index(cue)] - self.config.rest_cost * hazard

        return max(candidates, key=utility)

    def _render_report(self, clinician: dict, details: dict) -> dict:
        """Clinician report (short, plain) and system details (technical)."""
        return {
            "text": CLINICIAN_REPORT_TEMPLATE.format(**clinician),
            "details": SYSTEM_DETAILS_TEMPLATE.format(**details),
        }

    # ------------------------------------------------------------------ window

    def resolve(
        self,
        *,
        report: PerceptionReport,
        perception: Proposal,
        planning: Proposal,
        verdict: SafetyVerdict,
    ) -> WindowDecision:
        posteriors = planning.details["posteriors"]
        plan = list(planning.action)

        def decision(final, status, released, selected, **extra) -> WindowDecision:
            confidence = [float(posteriors[h][PRIMITIVES.index(cue)]) for h, cue in enumerate(final)]
            return WindowDecision(
                window_index=report.window_index,
                target_frame_index=report.target_frame_index,
                final_action=list(final),
                confidence=confidence,
                current_primitive=final[0],
                next_primitive=final[-1],
                next_confidence=confidence[-1],
                status=status,
                released=released,
                selected_agent=selected,
                **extra,
            )

        if verdict.verdict in ("block", "escalate"):
            return decision(plan, verdict.verdict, False, "safety")

        if verdict.verdict == "revise":
            chosen = decision(list(verdict.revised_action or plan), "revise", True, "safety")
        else:
            model = list(perception.action)
            final = []
            for h in range(len(plan)):
                cue = plan[h]
                if model[h] != plan[h]:
                    candidate = self.tools("select_cue", candidates=[plan[h], model[h]], posterior=posteriors[h])
                    if candidate != plan[h] and self.safety.cue_permitted(candidate, posteriors[h]):
                        cue = candidate
                final.append(cue)
            selected = "planning" if final == plan else "perception" if final == model else "coordinator"
            chosen = decision(final, "approve", True, selected, disagreement=model != plan)

        violation = self._first_violation(chosen, posteriors)
        if violation is not None:
            chosen = chosen.model_copy(update={"coordination_failure": f"released_cue_not_permitted_h{violation}"})
        return chosen

    def _first_violation(self, decision: WindowDecision, posteriors) -> int | None:
        for h, cue in enumerate(decision.final_action):
            if not self.safety.cue_permitted(cue, posteriors[h]):
                return h
        return None

    @staticmethod
    def feedback(decision: WindowDecision, verdict: SafetyVerdict | None) -> str:
        if decision.status == "block":
            return "Stopped: input unusable for this window."
        if decision.status == "escalate":
            reason = verdict.rule_hits[0] if verdict and verdict.rule_hits else decision.coordination_failure
            return f"Hold: clinician review requested ({reason})."
        lead = (len(decision.final_action) - 1) / STREAM_FPS
        prefix = "Adjusted: " if decision.status == "revise" else ""
        return (
            f"{prefix}current {decision.current_primitive} ({decision.confidence[0]:.2f}) -> "
            f"future (+{lead:.2f} s) {decision.next_primitive} ({decision.next_confidence:.2f})"
        )

    # ---------------------------------------------------------------- exercise

    def exercise_decision(self, status_counts: dict[str, int], n_windows: int, blocked: bool, clinician: bool) -> str:
        if blocked:
            return "block"
        withheld = status_counts.get("escalate", 0) + status_counts.get("block", 0)
        if clinician or (n_windows and withheld / n_windows > self.config.exercise_escalate_fraction):
            return "escalate"
        return "revise" if status_counts.get("revise", 0) else "approve"

    def render_report(self, clinician: dict, details: dict) -> dict:
        return self.tools("render_report", clinician=clinician, details=details)
