"""Planning agent: the primitive to do now (frame t) and next (frame t+4). Deterministic.

Each window the model gives a distribution for frames t..t+4. Planning fuses it with
what came before, causally:

    belief_h ∝ q_h ** temperature * predicted_h ** transition_strength

where predicted_h comes from the previous belief through a frame-level transition
matrix that mixes population statistics with the patient's own history,
`A = norm(kappa * A_pop + C_patient)`, and the belief at t+4 carries into the next
window. At the first frame of an exercise there is no previous belief and the
prediction is uniform (`START_BELIEF`). The headline recommendation is the fused
primitive at t+4.

At the defaults (transition_strength = 0, temperature = 1) the plan equals the model's own prediction, so planning cannot make the cue worse than the
model; `multi_agents.tune` raises the strengths only where validation subjects gain.
"""

from __future__ import annotations

import numpy as np

from multi_agents.priors import PopulationPriors
from multi_agents.settings import NUM_PRIMITIVES, PRIMITIVES, MASConfig
from multi_agents.state import PerceptionReport, Proposal
from multi_agents.tools import ToolBox, make_tool

# Prediction for the first frame of an exercise (no carried belief yet).
START_BELIEF = np.full(NUM_PRIMITIVES, 1.0 / NUM_PRIMITIVES)


def blended_transition(priors: PopulationPriors, patient_counts: np.ndarray, strength: float) -> np.ndarray:
    blended = strength * priors.transition + patient_counts
    return blended / blended.sum(axis=1, keepdims=True)


def filter_window(
    probs: np.ndarray,
    carry: np.ndarray | None,
    transition: np.ndarray,
    config: MASConfig,
    usable: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Posteriors for frames t..t+4 and the carry (the t+4 belief) for the next window."""
    previous = carry
    posteriors = []
    for row in probs:
        predicted = START_BELIEF if previous is None else previous @ transition
        if usable:
            emission = np.power(np.clip(row, 1e-8, None), config.emission_temperature)
            belief = emission * np.power(np.clip(predicted, 1e-12, None), config.transition_strength)
        else:
            belief = predicted
        belief = belief / belief.sum()
        posteriors.append(belief)
        previous = belief
    return np.asarray(posteriors), previous


class PlanningAgent:
    name = "planning"

    def __init__(self, priors: PopulationPriors, config: MASConfig) -> None:
        self.priors = priors
        self.config = config
        self.tools = ToolBox(
            self.name,
            [
                make_tool(
                    self._get_patient_primitive_memory,
                    "get_patient_primitive_memory",
                    "Patient-specific transition counts from history.",
                ),
                make_tool(
                    self._get_transition_prior,
                    "get_transition_prior",
                    "Frame-level transition matrix: population prior blended with patient counts.",
                ),
                make_tool(
                    self._forward_filter,
                    "forward_filter",
                    "Causal filter of the model's five per-frame posteriors (t..t+4).",
                ),
            ],
        )

    # ------------------------------------------------------------------ tools

    def _get_patient_primitive_memory(self, primitive_memory: dict) -> dict:
        counts = np.asarray(primitive_memory["transition_counts"], dtype=np.float64)
        strength = self.config.population_prior_strength
        row_totals = counts.sum(axis=1)
        return {
            "transition_counts": counts.tolist(),
            "patient_weight": float(np.mean(row_totals / (row_totals + strength))),
        }

    def _get_transition_prior(self, patient_transition_counts: list[list[float]]) -> list[list[float]]:
        return blended_transition(
            self.priors, np.asarray(patient_transition_counts), self.config.population_prior_strength
        ).tolist()

    def _forward_filter(
        self,
        probs: list[list[float]],
        transition: list[list[float]],
        carry: list[float] | None = None,
        usable: bool = True,
    ) -> dict:
        posteriors, carry_out = filter_window(
            np.asarray(probs),
            None if carry is None else np.asarray(carry),
            np.asarray(transition),
            self.config,
            usable,
        )
        return {"posteriors": posteriors.tolist(), "carry": carry_out.tolist()}

    # ---------------------------------------------------------------- proposals

    def plan(
        self,
        report: PerceptionReport,
        carry: list[float] | None,
        primitive_memory: dict,
        task_name: str,
    ) -> Proposal:
        memory = self.tools("get_patient_primitive_memory", primitive_memory=primitive_memory)
        transition = self.tools("get_transition_prior", patient_transition_counts=memory["transition_counts"])
        filtered = self.tools(
            "forward_filter",
            probs=[[row[name] for name in PRIMITIVES] for row in report.probs],
            transition=transition,
            carry=carry,
            usable=report.integrity_ok,
        )
        posteriors = np.asarray(filtered["posteriors"])
        plan_ids = posteriors.argmax(axis=1)
        model_ids = np.asarray([PRIMITIVES.index(label) for label in report.predicted_labels])
        rows = np.arange(len(plan_ids))
        benefit = float(np.mean(posteriors[rows, plan_ids] - posteriors[rows, model_ids]))
        changed = int(np.sum(plan_ids != model_ids))
        constraints = []
        if not report.integrity_ok:
            constraints.append("no_observation_prior_only")
        if memory["patient_weight"] > 0.5 and self.config.transition_strength > 0:
            constraints.append("patient_history_dominates_prior")
        action = [PRIMITIVES[i] for i in plan_ids]
        return Proposal(
            agent=self.name,
            action=action,
            confidence=float(posteriors.max(axis=1).mean()),
            expected_benefit=benefit,
            constraints=constraints,
            rationale=(
                f"Current {action[0]}, future {action[-1]}; fused the model with the {task_name} "
                f"transition prior (patient weight {memory['patient_weight']:.2f}); changed {changed}/5 frames."
            ),
            source="rule",
            details={
                "posteriors": filtered["posteriors"],
                "carry": filtered["carry"],
                "horizon_confidence": posteriors.max(axis=1).tolist(),
                "patient_weight": memory["patient_weight"],
            },
        )
