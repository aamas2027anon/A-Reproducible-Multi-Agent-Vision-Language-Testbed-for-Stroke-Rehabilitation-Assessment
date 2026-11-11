from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from multi_agents import metrics
from multi_agents.report import MODES, ablation_table, export_report as export_mas_report
from multi_agents.settings import MASConfig
from config import VARIANTS
from config import RuntimeConfig
from train.report import collect_seed_metrics
from sign_flip import sign_flip_pvalue


def frames() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": subject, "clip_id": f"{subject}_drinking1_1", "horizon": h,
         "absolute_frame_index": 115 + h, "actual_label_id": h, "predicted_label_id": h,
         "model_label_id": h, "plan_label_id": h, "released": True}
        for subject in ("s04", "s17") for h in range(5)
    ])


class MeasurementTests(unittest.TestCase):
    def test_collapsed_sequences_and_aer_can_exceed_one(self):
        frame = frames().iloc[:4].copy()
        frame["actual_label_id"] = [0, 0, 0, 0]
        frame["predicted_label_id"] = [0, 1, 0, 1]
        self.assertEqual(metrics.sequence_scores(frame), {"edit_score": 0.25, "action_error_rate": 3.0})
        frame["predicted_label_id"] = [0, 0, 0, 0]
        self.assertEqual(metrics.sequence_scores(frame)["edit_score"], 1)

    def test_counts_restart_at_clip_and_frame_gaps(self):
        frame = frames().iloc[:4].copy()
        frame[["actual_label_id", "predicted_label_id"]] = 0
        frame["clip_id"] = ["a", "a", "b", "b"]
        frame["absolute_frame_index"] = [115, 116, 115, 117]
        counts = metrics.subject_primitive_counts(frame)
        self.assertEqual(counts.loc[counts.primitive == "reach", "actual_count"].item(), 3)
        self.assertEqual(counts.loc[counts.primitive == "total", "predicted_count"].item(), 3)

    def test_uefma_running_mean_uses_exercise_order_and_subject_boundaries(self):
        records = pd.DataFrame([
            {"subject_id": "s04", "clip_id": "b", "exercise_index": 1, "model_ue_fma": 60},
            {"subject_id": "s04", "clip_id": "a", "exercise_index": 0, "model_ue_fma": 20},
            {"subject_id": "s04", "clip_id": "a", "exercise_index": 0, "model_ue_fma": 40},
            {"subject_id": "s17", "clip_id": "c", "exercise_index": 0, "model_ue_fma": 10},
        ])
        source = SimpleNamespace(true_uefma=lambda subject: 40)
        exercise = metrics.exercise_frame(records, source, None, MASConfig())
        self.assertEqual(exercise.ue_model_running_mean.tolist(), [30, 45, 10])
        self.assertEqual(exercise.ue_abs_error_running_mean.tolist(), [10, 5, 30])
        patients = metrics.uefma_subject_frame(records, exercise, source, "vlm", MASConfig())
        self.assertEqual(patients.loc[patients.subject_id == "s04", "predicted_ue_fma"].item(), 40)

    def test_withheld_predictions_count_in_classification_not_unsafe_denominator(self):
        frame = frames()
        frame.loc[[2, 7], "predicted_label_id"] = 0  # active on true rest
        frame.loc[7, "released"] = False
        counts = metrics.subject_primitive_counts(frame)
        score = metrics.summary_metrics(
            mode="mas", primitives=frame, records=pd.DataFrame({"coordination_failure": [None]}),
            exercises=pd.DataFrame({"ue_abs_error_estimate": [2, 4], "ue_abs_error_running_mean": [1, 3]}),
            uefma_subjects=pd.DataFrame({"actual_ue_fma": [30, 50], "predicted_ue_fma": [32, 46]}),
            counts=counts, config=MASConfig(), priors=SimpleNamespace(transition=np.ones((5, 5)) / 5),
        )
        self.assertEqual(score["accuracy"], 0.8)
        self.assertAlmostEqual(score["unsafe_rate"], 1 / 9)
        self.assertEqual(score["coverage"], 0.9)
        self.assertEqual(score["ue_fma_exercise_mae"], 3)
        self.assertEqual(score["ue_fma_running_mean_mae"], 2)

    def test_reference_join_counts_and_rejects_duplicates(self):
        frame = frames()
        reference = frame.iloc[:-1].copy()
        reference.loc[0, "predicted_label_id"] = 1
        result = metrics.reference_agreement(frame, reference)
        self.assertEqual(result["matched_frames"], 9)
        self.assertEqual(result["replay_frames"], 10)
        self.assertAlmostEqual(result["agreement"], 8 / 9)
        with self.assertRaises(pd.errors.MergeError):
            metrics.reference_agreement(pd.concat([frame, frame.iloc[:1]]), reference)

    def test_masked_targets_are_not_scored(self):
        record = {"subject_id": "s04", "clip_id": "clip", "window_index": 0,
                  "target_frame_index": 115, "model_action": ["rest"] * 5,
                  "plan_action": ["rest"] * 5, "final_action": ["rest"] * 5, "released": False}
        source = SimpleNamespace(find=lambda *args: None, labels=lambda exercise: np.array([[2, 2, 5, 5, 5]]))
        scored = metrics.primitive_frame(pd.DataFrame([record]), source)
        self.assertEqual(scored.horizon.tolist(), [0, 1])

    def test_sign_flips_are_reproducible_and_two_sided(self):
        p = sign_flip_pvalue([1, 1], seed=7)
        self.assertAlmostEqual(p, 0.5, delta=0.03)
        self.assertEqual(p, sign_flip_pvalue([-1, -1], seed=7))
        self.assertEqual(sign_flip_pvalue([0, 0]), 1)


class ReportTests(unittest.TestCase):
    def make_inputs(self, root):
        training, mas = root / "training", root / "mas"
        for seed in (1337, 9575):
            for run, _ in VARIANTS:
                predictions = training / "seeds" / f"seed_{seed}" / run / "predictions"
                predictions.mkdir(parents=True)
                frames().to_csv(predictions / "test_combined_window_predictions.csv", index=False)
                pd.DataFrame({"subject_id": ["s04", "s17"], "actual_ue_fma": [30, 55],
                              "predicted_ue_fma": [32 + seed % 3, 50 - seed % 2]}).to_csv(
                    predictions / "test_ue_fma_subject_predictions.csv", index=False)
            directory = mas / f"seed_{seed}"
            evaluation = directory / "eval_test"
            evaluation.mkdir(parents=True)
            (directory / "tuned_config.json").write_text(json.dumps(MASConfig().to_dict()))
            rows, counts, changes, predictions = [], [], [], []
            for mode in MODES:
                frame = frames()
                score = metrics.summary_metrics(
                    mode=mode, primitives=frame,
                    records=pd.DataFrame({"coordination_failure": [None]}),
                    exercises=pd.DataFrame({"ue_abs_error_estimate": [2, 4], "ue_abs_error_running_mean": [1, 3]}),
                    uefma_subjects=pd.DataFrame({"actual_ue_fma": [30, 55], "predicted_ue_fma": [32, 50]}),
                    counts=metrics.subject_primitive_counts(frame), config=MASConfig(),
                    priors=SimpleNamespace(transition=np.ones((5, 5)) / 5),
                )
                rows.append(score)
                counts.append(metrics.subject_primitive_counts(frame).assign(mode=mode))
                predictions.append(frame.assign(mode=mode))
                if mode != "vlm":
                    changes.extend({"mode": mode, **row} for row in metrics.change_attribution(frame))
            pd.DataFrame(rows).to_csv(evaluation / "metrics.csv", index=False)
            pd.concat(counts).to_csv(evaluation / "count_errors.csv", index=False)
            pd.DataFrame(changes).to_csv(evaluation / "change_attribution.csv", index=False)
            pd.concat(predictions).to_csv(evaluation / "primitive_predictions.csv", index=False)
        return training, mas

    def test_report_command_works_without_training_or_inference_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, mas = self.make_inputs(root)
            output = root / "report"
            command = '''import importlib.abc, runpy, sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'lightning', 'peft', 'training'} or fullname == 'multi_agents.inference':
            raise AssertionError('Reporting imported ' + fullname)
sys.meta_path.insert(0, BlockTraining())
runpy.run_module('report', run_name='__main__')
'''
            result = subprocess.run([sys.executable, "-c", command, "--training-root", str(training),
                                     "--mas-root", str(mas), "--output-dir", str(output)],
                                    cwd=Path(__file__).parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 2)
            self.assertEqual({p.name for p in (output / "tables").glob("*.csv")}, {
                "vlm_seed_metrics.csv", "vlm_uefma.csv", "vlm_text_vs_base.csv", "vlm_prefix_vs_base.csv",
                "mas_seed_metrics.csv", "mas_vs_vlm.csv", "tuned_settings.csv", "mas_additional_metrics.csv",
                "primitive_counts.csv", "agent_diagnostics.csv", "uefma_aggregation.csv", "prediction_agreement.csv",
                "no_memory_vs_mas.csv", "no_planning_vs_mas.csv", "no_safety_vs_mas.csv",
            })
            table = pd.read_csv(output / "tables" / "mas_vs_vlm.csv")
            self.assertTrue(table["n seeds"].eq(2).all())
            agreement = pd.read_csv(output / "tables" / "prediction_agreement.csv")
            self.assertTrue(agreement.agreement.eq(1).all())

    def test_partial_seed_set_and_missing_agreement_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, mas = self.make_inputs(root)
            import shutil
            shutil.rmtree(mas / "seed_1337")
            (mas / "seed_9575" / "eval_test" / "primitive_predictions.csv").unlink()
            result = export_mas_report(mas, root / "report", training)
            self.assertTrue(pd.read_csv(result["mas_vs_vlm"])["n seeds"].eq(1).all())
            self.assertEqual(pd.read_csv(result["prediction_agreement"]).status.item(), "unavailable")

    def test_missing_mode_and_duplicate_pair_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, mas = self.make_inputs(root)
            path = mas / "seed_1337" / "eval_test" / "metrics.csv"
            original = pd.read_csv(path)
            original.iloc[:-1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing modes"):
                export_mas_report(mas, root / "report", training)
            pd.concat([original, original.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate seed/mode"):
                export_mas_report(mas, root / "report", training)

    def test_conditioning_inputs_must_score_matching_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, _ = self.make_inputs(root)
            path = training / "seeds" / "seed_9575" / VARIANTS[-1][0] / "predictions" / "test_combined_window_predictions.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "actual_label_id"] = 4
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "different frames or labels"):
                collect_seed_metrics(RuntimeConfig(root, training, root))

    def test_ablation_direction_and_sample_sd(self):
        rows = []
        from multi_agents.report import ABLATION_METRICS
        for seed in (1, 2):
            rows.extend({"seed": seed, "mode": mode,
                         **{column: float(seed) + (1 if mode == "mas" else 0)
                            for column, _ in ABLATION_METRICS}}
                        for mode in ("mas", "mas_no_memory"))
        table = ablation_table(pd.DataFrame(rows), "mas_no_memory").set_index("Metric")
        self.assertEqual(table.loc["Count MAE", "Better"], 2)
        self.assertEqual(table.loc["Accuracy (frame)", "Worse"], 2)
        self.assertAlmostEqual(table.loc["Count MAE", "MAS SD"], np.sqrt(0.5))


if __name__ == "__main__":
    unittest.main()
