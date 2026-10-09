from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from config import CONDITIONING_PREFIX, RuntimeConfig, VARIANTS
from summary import export_all_runs_summary


class SummaryTests(unittest.TestCase):
    @staticmethod
    def _write_run(
        run_dir: Path, run_name: str, conditioning: str, fold_id: int, split: str
    ) -> None:
        prediction_dir = run_dir / "predictions"
        prediction_dir.mkdir(parents=True)
        subject = f"s{fold_id + 1:02d}" if fold_id >= 0 else "t01"
        combined = pd.DataFrame(
            [
                {
                    "subject_id": subject,
                    "task_name": "tabletop",
                    "clip_id": "clip",
                    "target_frame_index": 115,
                    "absolute_frame_index": 115 + horizon,
                    "horizon": horizon,
                    "prediction_position": "current" if horizon == 0 else f"future_{horizon}",
                    "actual_label_id": horizon,
                    "predicted_label_id": horizon,
                    "actual_primitive": label,
                    "predicted_primitive": label,
                    "prediction_probability": 1.0,
                    "actual_ue_fma": 40.0,
                    "predicted_ue_fma": 39.0,
                    "run_name": run_name,
                    "fold_id": fold_id,
                    "split": split,
                }
                for horizon, label in enumerate(
                    ("reach", "reposition", "rest", "stabilize", "transport")
                )
            ]
        )
        combined.to_csv(
            prediction_dir / f"{split}_combined_window_predictions.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "subject_id": subject,
                    "actual_ue_fma": 40.0,
                    "predicted_ue_fma": 39.0,
                    "absolute_error": 1.0,
                    "run_name": run_name,
                    "fold_id": fold_id,
                    "split": split,
                }
            ]
        ).to_csv(
            prediction_dir / f"{split}_ue_fma_subject_predictions.csv", index=False
        )
        (prediction_dir / "prediction_manifest.json").write_text(
            json.dumps({"prediction_complete": True}), encoding="utf-8"
        )
        manifest = {
            "training_complete": True,
            "config_id": "test",
            "run_name": run_name,
            "conditioning": conditioning,
            "use_patient_prefix": conditioning == CONDITIONING_PREFIX,
            "max_epochs": 5,
            "epochs": 5,
            "completed_epochs": 5,
            "resumed_from_checkpoint": False,
            "reused_completed_checkpoint": False,
            "early_stopping_enabled": True,
            "early_stopped": False,
            "stop_reason": "max_epochs",
            "early_stopping_monitor": "val_joint_loss",
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.01,
            "early_stopping_best_score": 0.5,
            "early_stopping_best_epoch": 4,
            "early_stopping_stopped_after_epoch": None,
            "final_metrics": {},
            "seed": 90998 + max(fold_id, 0),
            "train_subjects": ["s31"],
            "validation_subjects": ["s32"],
            "prediction_subjects": [subject],
            "final_checkpoint": str(run_dir / "checkpoints" / "last.ckpt"),
            "history_path": str(run_dir / "logs" / "metrics.csv"),
        }
        (run_dir / "training_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_all_runs_summary_uses_combined_prediction_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                data_root=root / "data",
                output_root=root / "outputs",
                report_root=root / "reports",
                cache_root=root / "cache",
                n_folds=4,
            )
            for fold_id in range(config.n_folds):
                for run_name, conditioning in VARIANTS:
                    self._write_run(
                        config.output_root
                        / "folds"
                        / f"fold_{fold_id:02d}"
                        / run_name,
                        run_name,
                        conditioning,
                        fold_id,
                        "oof",
                    )
            for run_name, conditioning in VARIANTS:
                self._write_run(
                    config.output_root / "full_train_test" / run_name,
                    run_name,
                    conditioning,
                    -1,
                    "test",
                )
            output = export_all_runs_summary(config)
            frame = pd.read_csv(output)
            self.assertEqual(len(frame), len(VARIANTS) * (config.n_folds + 1))
            self.assertEqual(
                set(frame["conditioning"]), {mode for _, mode in VARIANTS}
            )
            self.assertTrue(
                {"primitive_all_accuracy", "ue_fma_mae", "count_mae"}.issubset(
                    frame.columns
                )
            )
            self.assertTrue(frame["primitive_all_accuracy"].eq(1.0).all())


if __name__ == "__main__":
    unittest.main()
