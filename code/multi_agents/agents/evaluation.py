"""Evaluation agent: reports UE-FMA and maintains the patient's estimate (deterministic).

The model predicts a UE-FMA score per window. An exercise's score is the mean of its
windows, as in the training evaluation. Across exercises the agent keeps a
precision-weighted estimate for the patient:

* each exercise is an observation with variance s^2 / n_eff + sigma_between^2, where
  n_eff discounts the heavy overlap of consecutive 24-frame windows;
* an exercise further than k sigma from the estimate is Huber down-weighted and
  flagged as drift (safety rule E2) once enough exercises exist;
* an exercise without usable windows leaves the estimate unchanged ("hold").

Pooling (mu, sigma^2, the weights and the drift rule) runs on the model's raw scale.
On output only, the estimate is linearly calibrated (fitted on validation by tune.py):

    mu_cal = clip(ue_calib_a + ue_calib_b * mu, 0, 66),   sigma_cal = ue_calib_b * sigma

mu_cal is stored next to mu in the UE-FMA memory, and it sets the impairment band
(provisional and after the exercise), the clinician report and every exported estimate.

UE-FMA is constant per subject in this data, so the gain over the single model is
variance reduction across repeated exercises.
"""

from __future__ import annotations

import math

import numpy as np

from multi_agents.settings import MASConfig
from multi_agents.state import Proposal
from multi_agents.tools import ToolBox, make_tool


class EvaluationAgent:
    name = "evaluation"

    def __init__(self, config: MASConfig) -> None:
        self.config = config
        self.tools = ToolBox(
            self.name,
            [
                make_tool(self._get_uefma_history, "get_uefma_history", "The patient's UE-FMA estimate and history."),
                make_tool(
                    self._update_uefma_estimate,
                    "update_uefma_estimate",
                    "Fold one exercise's window scores into the patient's UE-FMA estimate.",
                ),
            ],
        )

    def calibrated(self, ue_fma_memory: dict) -> tuple[float | None, float | None]:
        """(mu_cal, sigma_cal) of the patient estimate, computed from the raw mu and sigma^2."""
        variance = ue_fma_memory["variance"]
        sd = None if variance is None else self.config.ue_calib_b * math.sqrt(variance)
        return self.config.calibrate_uefma(ue_fma_memory["mean"]), sd

    def _get_uefma_history(self, ue_fma_memory: dict) -> dict:
        return {
            "mean": ue_fma_memory["mean"],
            "mean_cal": self.config.calibrate_uefma(ue_fma_memory["mean"]),
            "variance": ue_fma_memory["variance"],
            "n_exercises": len(ue_fma_memory["exercise_estimates"]),
            "band": ue_fma_memory["band"],
        }

    def _exercise_observation(self, window_scores: list[float]) -> tuple[float, float]:
        scores = np.asarray(window_scores, dtype=np.float64)
        n_eff = max(1.0, len(scores) / self.config.uefma_window_decorrelation)
        within = float(scores.var(ddof=1)) if len(scores) > 1 else 0.0
        return float(scores.mean()), within / n_eff + self.config.uefma_between_exercise_sd**2

    def _update_uefma_estimate(self, ue_fma_memory: dict, window_scores: list[float], exercise_index: int) -> dict:
        memory = dict(ue_fma_memory)
        if not window_scores:
            # Hold: mu and sigma^2 are unchanged; mu_cal and the band follow the current calibration.
            if memory["mean"] is not None:
                mean_cal = self.config.calibrate_uefma(memory["mean"])
                memory.update(mean_cal=mean_cal, band=self.config.band(mean_cal))
            return {"memory": memory, "decision": "hold", "exercise_score": None, "drift": False}
        score, variance = self._exercise_observation(window_scores)
        weight, decision, drift = 1.0 / variance, "accept", False
        if memory["mean"] is not None:
            scale = math.sqrt(variance + memory["variance"])
            residual = abs(score - memory["mean"])
            enough = len(memory["exercise_estimates"]) >= self.config.uefma_min_exercises_for_drift
            if enough and residual > self.config.uefma_drift_sigma * scale:
                weight *= self.config.uefma_drift_sigma * scale / residual
                decision, drift = "down-weight", True
            precision = 1.0 / memory["variance"] + weight
            mean = (memory["mean"] / memory["variance"] + weight * score) / precision
            new_variance = 1.0 / precision
        else:
            mean, new_variance = score, variance
        mean_cal = self.config.calibrate_uefma(mean)
        memory.update(
            exercise_estimates=[*memory["exercise_estimates"], score],
            exercise_weights=[*memory["exercise_weights"], weight],
            mean=float(mean),
            mean_cal=mean_cal,
            variance=float(new_variance),
            band=self.config.band(mean_cal),
            drift_exercises=[*memory["drift_exercises"], exercise_index] if drift else list(memory["drift_exercises"]),
        )
        return {"memory": memory, "decision": decision, "exercise_score": score, "drift": drift}

    def provisional(self, ue_fma_memory: dict, window_scores: list[float]) -> tuple[float | None, str]:
        """Calibrated estimate mu_cal and its band while an exercise is running (drives safety thresholds).

        The running exercise is pooled with the stored estimate on the raw scale, then calibrated.
        """
        if not window_scores:
            mean = ue_fma_memory["mean"]
        else:
            score, variance = self._exercise_observation(window_scores)
            if ue_fma_memory["mean"] is None:
                mean = score
            else:
                precision = 1.0 / ue_fma_memory["variance"] + 1.0 / variance
                mean = (ue_fma_memory["mean"] / ue_fma_memory["variance"] + score / variance) / precision
        mean_cal = self.config.calibrate_uefma(mean)
        return mean_cal, self.config.band(mean_cal)

    def assess(self, ue_fma_memory: dict, window_scores: list[float], exercise_index: int) -> tuple[Proposal, dict]:
        before = self.tools("get_uefma_history", ue_fma_memory=ue_fma_memory)
        result = self.tools(
            "update_uefma_estimate",
            ue_fma_memory=ue_fma_memory,
            window_scores=window_scores,
            exercise_index=exercise_index,
        )
        memory = result["memory"]
        mean_cal, _ = self.calibrated(memory)
        exercise_cal = self.config.calibrate_uefma(result["exercise_score"])
        if memory["variance"] is None:
            confidence, benefit = 0.0, 0.0
        else:
            confidence = 1.0 / (1.0 + math.sqrt(memory["variance"]))
            benefit = 1.0 if before["variance"] is None else 1.0 - memory["variance"] / before["variance"]
        proposal = Proposal(
            agent=self.name,
            action=result["decision"],
            confidence=confidence,
            expected_benefit=float(benefit),
            constraints=["uefma_drift"] if result["drift"] else [],
            rationale=(
                "No usable windows; estimate held."
                if result["exercise_score"] is None
                else f"Exercise UE-FMA {exercise_cal:.1f}; patient estimate {mean_cal:.1f} (calibrated)."
            ),
            source="rule",
            details={
                "exercise_score": exercise_cal,
                "exercise_score_raw": result["exercise_score"],
                "drift": result["drift"],
            },
        )
        return proposal, memory
