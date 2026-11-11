"""Window-level safety layer (R1 -> R2-escalate -> C1) and how the coordinator releases it.

    python -m unittest multi_agents.test_safety      (from the repository root)
"""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

from multi_agents.agents.coordinator import CoordinatorAgent
from multi_agents.agents.safety import SafetyAgent
from multi_agents.settings import HORIZONS, PRIMITIVES, MASConfig
from multi_agents.state import PerceptionReport, Proposal


def row(**probabilities: float) -> list[float]:
    """A belief over PRIMITIVES; unnamed classes share what is left equally."""
    rest = [name for name in PRIMITIVES if name not in probabilities]
    remainder = (1.0 - sum(probabilities.values())) / len(rest) if rest else 0.0
    return [probabilities.get(name, remainder) for name in PRIMITIVES]


def plan_from(posteriors: list[list[float]]) -> Proposal:
    """Planning's proposal: the arg-max of each frame's belief, as PlanningAgent.plan does."""
    action = [PRIMITIVES[max(range(len(r)), key=r.__getitem__)] for r in posteriors]
    return Proposal(
        agent="planning",
        action=action,
        confidence=0.5,
        expected_benefit=0.0,
        details={"posteriors": posteriors, "carry": posteriors[-1], "horizon_confidence": [max(r) for r in posteriors]},
    )


def report_for(posteriors: list[list[float]], integrity_ok: bool = True) -> PerceptionReport:
    probs = [dict(zip(PRIMITIVES, r)) for r in posteriors]
    labels = [PRIMITIVES[max(range(len(r)), key=r.__getitem__)] for r in posteriors]
    return PerceptionReport(
        window_index=0,
        target_frame_index=115,
        probs=probs,
        predicted_labels=labels,
        confidence=[max(r) for r in posteriors],
        ue_fma=40.0,
        integrity_ok=integrity_ok,
    )


def perception_from(report: PerceptionReport) -> Proposal:
    return Proposal(agent="perception", action=list(report.predicted_labels), confidence=0.5, expected_benefit=0.5, source="model")


class C1CostDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MASConfig()  # rest_cost = 0.5
        self.safety = SafetyAgent(self.config)

    def test_replaces_movement_with_rest(self) -> None:
        # reach: 0.45 - 0.5 * 0.40 = 0.25 < rest 0.40
        posteriors = [row(reach=0.45, rest=0.40)] * HORIZONS
        plan = plan_from(posteriors)
        self.assertEqual(plan.action, ["reach"] * HORIZONS)
        verdict = self.safety.check_window(report=report_for(posteriors), plan=plan, band="moderate")
        self.assertEqual(verdict.verdict, "revise")
        self.assertEqual(verdict.revised_action, ["rest"] * HORIZONS)
        self.assertTrue(all("C1" in hit for hit in verdict.rule_hits))

    def test_replaces_movement_with_stabilize(self) -> None:
        # reach: 0.40 - 0.5 * 0.15 = 0.325 < stabilize 0.35 (> rest 0.15)
        posteriors = [row(reach=0.40, stabilize=0.35, rest=0.15)] * HORIZONS
        verdict = self.safety.check_window(report=report_for(posteriors), plan=plan_from(posteriors), band="moderate")
        self.assertEqual(verdict.verdict, "revise")
        self.assertEqual(verdict.revised_action, ["stabilize"] * HORIZONS)

    def test_never_changes_a_non_movement_cue(self) -> None:
        for rest_cost in (0.0, 0.5, 3.0):
            safety = SafetyAgent(replace(self.config, rest_cost=rest_cost))
            for cue in ("rest", "stabilize"):
                posteriors = [row(**{cue: 0.40, "reach": 0.39})] * HORIZONS
                plan = plan_from(posteriors)
                self.assertEqual(plan.action, [cue] * HORIZONS)
                verdict = safety.check_window(report=report_for(posteriors), plan=plan, band="moderate")
                self.assertEqual(verdict.verdict, "approve", (rest_cost, cue))
                self.assertIsNone(verdict.revised_action)

    def test_ignores_transitions(self) -> None:
        # No transition input exists: neither the agent nor the window check takes priors or a previous cue.
        self.assertNotIn("priors", inspect.signature(SafetyAgent.__init__).parameters)
        for name in ("last_released", "previous", "active_streak"):
            self.assertNotIn(name, inspect.signature(SafetyAgent.check_window).parameters)
            self.assertNotIn(name, inspect.signature(SafetyAgent.cue_permitted).parameters)
        # Each frame cost-optimal, alternating transport and rest: approved, whatever the transitions.
        posteriors = [row(transport=0.9) if h % 2 == 0 else row(rest=0.9) for h in range(HORIZONS)]
        verdict = self.safety.check_window(report=report_for(posteriors), plan=plan_from(posteriors), band="moderate")
        self.assertEqual(verdict.verdict, "approve")
        self.assertEqual(verdict.rule_hits, [])


class R2EscalationTest(unittest.TestCase):
    def test_uses_the_band_threshold(self) -> None:
        config = MASConfig()  # escalate_confidence: severe 0.35, moderate 0.30, mild 0.25
        safety = SafetyAgent(config)
        # b_0(plan_0) = 0.32; reach stays cost-optimal (0.32 - 0.5 * 0.17 = 0.235 > 0.17)
        posteriors = [row(reach=0.32)] * HORIZONS
        plan = plan_from(posteriors)
        report = report_for(posteriors)
        expected = {"severe": "escalate", "moderate": "approve", "mild": "approve"}
        for band, verdict_name in expected.items():
            verdict = safety.check_window(report=report, plan=plan, band=band)
            self.assertEqual(verdict.verdict, verdict_name, band)
        escalated = safety.check_window(report=report, plan=plan, band="severe")
        self.assertTrue(escalated.rule_hits[0].startswith("R2"))


class R1BlockTest(unittest.TestCase):
    def test_blocks_unusable_input(self) -> None:
        config = MASConfig()
        safety = SafetyAgent(config)
        posteriors = [row(reach=0.9)] * HORIZONS
        report = report_for(posteriors, integrity_ok=False)
        plan = plan_from(posteriors)
        verdict = safety.check_window(report=report, plan=plan, band="moderate")
        self.assertEqual(verdict.verdict, "block")
        decision = CoordinatorAgent(config, safety).resolve(
            report=report, perception=perception_from(report), planning=plan, verdict=verdict
        )
        self.assertFalse(decision.released)
        self.assertEqual(decision.status, "block")


class CoordinatorReleaseTest(unittest.TestCase):
    def test_revise_is_released_unchanged(self) -> None:
        config = MASConfig()
        safety = SafetyAgent(config)
        coordinator = CoordinatorAgent(config, safety)
        posteriors = [row(reach=0.45, rest=0.40)] * 3 + [row(rest=0.9)] * (HORIZONS - 3)
        report = report_for(posteriors)
        plan = plan_from(posteriors)
        verdict = safety.check_window(report=report, plan=plan, band="moderate")
        self.assertEqual(verdict.verdict, "revise")
        decision = coordinator.resolve(report=report, perception=perception_from(report), planning=plan, verdict=verdict)
        self.assertEqual(decision.status, "revise")
        self.assertTrue(decision.released)
        self.assertEqual(decision.final_action, verdict.revised_action)
        self.assertEqual(decision.selected_agent, "safety")
        self.assertIsNone(decision.coordination_failure)


if __name__ == "__main__":
    unittest.main()
