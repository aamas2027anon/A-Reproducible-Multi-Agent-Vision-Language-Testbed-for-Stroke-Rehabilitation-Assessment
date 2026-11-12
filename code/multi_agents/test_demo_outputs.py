from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from multi_agents.demo_outputs import export_demo
from multi_agents.settings import MASConfig, PRIMITIVES
from multi_agents.state import ExerciseRecord, PatientMemory, PatientProfile


def exercise_record(index):
    return ExerciseRecord(
        exercise_index=index, clip_id="s04_drinking1_1", task_name="drinking", n_windows=1,
        status_counts={"approve": 1, "revise": 0, "escalate": 0, "block": 0}, released_windows=1,
        mean_confidence=0.8, ue_fma_exercise=35, ue_fma_estimate_after=40, ue_fma_ci95=3,
        ue_fma_update="accept", predicted_counts={name: 1 for name in PRIMITIVES},
        disagreements=1, revisions=0, coordination_failures=0, latency_ms={}, clinician_flag=False,
    )


class DemoOutputTests(unittest.TestCase):
    def test_agent_cycle_emits_full_trace_and_preview_keeps_original_memory(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from multi_agents.agents import CoordinatorAgent, EvaluationAgent, PerceptionAgent, PlanningAgent, SafetyAgent
        from multi_agents.graph import Services, build_graph
        from multi_agents.inference import ExerciseInference
        from multi_agents.memory import WindowLog, delete_thread, load_patient
        from multi_agents.priors import PopulationPriors
        from multi_agents.session import RehabSession

        config = MASConfig()
        result = ExerciseInference(
            probs=np.array([[[0.7, 0.1, 0.1, 0.05, 0.05]] * 5]), ue_fma=np.array([40.0]),
            target_frames=np.array([115]), window_ok=np.array([True]), latency_ms=0.0,
        )
        priors = PopulationPriors(np.ones((5, 5)) / 5, np.ones(5) / 5, ("s01",))
        safety = SafetyAgent(config)
        services = Services(config, PerceptionAgent(SimpleNamespace(run=lambda key: result)),
                            PlanningAgent(priors, config), safety, EvaluationAgent(config),
                            CoordinatorAgent(config, safety), WindowLog("test"), "mas")
        checkpointer = InMemorySaver()
        graph = build_graph(services, checkpointer)
        seed = {"profile": PatientProfile(subject_id="s04").model_dump()}
        exercise = SimpleNamespace(key="s04/clip", subject_id="s04", clip_id="clip", task_name="drinking")
        report = RehabSession(graph, services, "s04", seed).replay(exercise)
        trace = services.log.records[0]
        self.assertEqual(len(trace["model_probs"]), 5)
        np.testing.assert_allclose(trace["plan_beliefs"], result.probs[0])
        self.assertEqual(trace["estimated_uefma"], 40)
        self.assertEqual(trace["band"], "moderate")
        self.assertEqual(trace["safety_verdict"], "approve")
        self.assertTrue(trace["released"])
        self.assertIn("Clinician review:", report["text"])
        persistent = load_patient(graph, "s04")
        RehabSession(graph, services, "preview", seed, patient=persistent.model_dump()).replay(exercise)
        self.assertEqual(len(load_patient(graph, "s04").exercises), 1)
        self.assertEqual(len(load_patient(graph, "preview").exercises), 2)
        delete_thread(checkpointer, "preview")

    def test_preview_exports_result_and_persistent_memory_separately(self):
        source = SimpleNamespace(find=lambda *args: None, labels=lambda exercise: np.array([[0, 1, 2, 3, 4]]),
                                 true_uefma=lambda subject: 42)
        record = {
            "subject_id": "s04", "clip_id": "s04_drinking1_1", "window_index": 0,
            "target_frame_index": 115, "model_action": list(PRIMITIVES), "plan_action": list(PRIMITIVES),
            "final_action": list(PRIMITIVES), "released": False, "status": "escalate",
            "model_probs": [{name: 0.2 for name in PRIMITIVES}] * 5,
            "plan_beliefs": [[0.2] * 5] * 5, "estimated_uefma": 40, "band": "moderate",
            "rule_hits": ["R2"], "selected_agent": "planning",
        }
        persistent = PatientMemory(profile=PatientProfile(subject_id="s04"), exercises=[exercise_record(0)])
        result = persistent.model_copy(update={"exercises": [*persistent.exercises, exercise_record(1)]})
        before = persistent.model_dump()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            export_demo(output, source, [record], result, persistent, MASConfig(), "Exact prompt", True)
            details = json.loads((output / "run_details.json").read_text())
            self.assertFalse(details["persistent_memory_updated"])
            self.assertEqual(len(json.loads((output / "patient_memory.json").read_text())["exercises"]), 1)
            self.assertEqual(len(json.loads((output / "preview_patient_memory.json").read_text())["exercises"]), 2)
            traces = pd.read_csv(output / "window_decisions.csv")
            self.assertEqual(json.loads(traces.rule_hits.item()), ["R2"])
            self.assertEqual(len(json.loads(traces.plan_beliefs.item())), 5)
            for name in ("decision_timeline.png", "patient_trajectory.png"):
                self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG"))
            self.assertEqual((output / "model_prompt.txt").read_text(), "Exact prompt\n")
            export_demo(output, source, [record], persistent, persistent, MASConfig(), "Exact prompt", False)
            self.assertFalse((output / "preview_patient_memory.json").exists())
            self.assertFalse((output / "patient_trajectory.png").exists())
        self.assertEqual(persistent.model_dump(), before)


if __name__ == "__main__":
    unittest.main()
