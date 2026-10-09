from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from config import POSITION_NAMES
from epoch_metrics import (
    EPOCH_METRIC_COLUMNS,
    EpochMetricsCallback,
    atomic_write_text,
    read_lightning_history,
    task_metric_rows,
)


LABELS = ("reach", "reposition", "rest", "stabilize", "transport")


def _combined_frame() -> pd.DataFrame:
    rows = []
    for subject_index in range(1, 4):
        for horizon, actual in enumerate(LABELS):
            predicted = actual if horizon % 2 == 0 else LABELS[(horizon + 1) % len(LABELS)]
            rows.append(
                {
                    "subject_id": f"t{subject_index:02d}",
                    "clip_id": f"t{subject_index:02d}_tabletop_0",
                    "task_name": "tabletop",
                    "target_frame_index": 115,
                    "horizon": horizon,
                    "absolute_frame_index": 115 + horizon,
                    "prediction_position": POSITION_NAMES[horizon],
                    "actual_primitive": actual,
                    "predicted_primitive": predicted,
                    "actual_label_id": LABELS.index(actual),
                    "predicted_label_id": LABELS.index(predicted),
                }
            )
    return pd.DataFrame(rows)


class EpochMetricsWriterTests(unittest.TestCase):
    def test_history_is_collapsed_to_one_row_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            # Lightning writes train and validation metrics on separate rows.
            pd.DataFrame(
                [
                    {"epoch": 0, "train_joint_loss": 1.0, "val_joint_loss": None},
                    {"epoch": 0, "train_joint_loss": None, "val_joint_loss": 1.2},
                    {"epoch": 1, "train_joint_loss": 0.8, "val_joint_loss": None},
                    {"epoch": 1, "train_joint_loss": None, "val_joint_loss": 1.1},
                ]
            ).to_csv(path, index=False)
            history = read_lightning_history(path)
            self.assertEqual(list(history["epoch"]), [0, 1])
            self.assertAlmostEqual(float(history.loc[0, "train_joint_loss"]), 1.0)
            self.assertAlmostEqual(float(history.loc[1, "val_joint_loss"]), 1.1)

    def test_missing_history_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history = read_lightning_history(Path(temporary) / "absent.csv")
            self.assertTrue(history.empty)

    def test_task_rows_cover_every_horizon(self) -> None:
        rows = task_metric_rows(_combined_frame(), 3, "run", "text", 42)
        self.assertEqual(len(rows), len(POSITION_NAMES))
        self.assertTrue(all(row["epoch"] == 3 for row in rows))
        self.assertEqual(
            {row["prediction_position"] for row in rows}, set(POSITION_NAMES)
        )
        self.assertTrue(all(row["conditioning"] == "text" for row in rows))

    def test_atomic_write_leaves_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "epoch_metrics.csv"
            atomic_write_text(path, "a,b\n1,2\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "a,b\n1,2\n")
            self.assertEqual(list(path.parent.glob(".*partial")), [])

    def test_persist_grows_by_one_row_per_epoch_and_writes_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            callback = EpochMetricsCallback.__new__(EpochMetricsCallback)
            callback.dataset = SimpleNamespace(subjects={"t01", "t02", "t03"})
            callback.run_name = "qwen_with_patient_text"
            callback.conditioning = "text"
            callback.seed = 9575
            callback.split = "test"
            callback.epoch_dir = root / "epochs"
            callback.metrics_path = callback.epoch_dir / "epoch_metrics.csv"
            callback.task_metrics_path = callback.epoch_dir / "epoch_task_metrics.csv"
            callback.manifest_path = callback.epoch_dir / "epoch_manifest.json"
            callback.plot_dir = callback.epoch_dir / "plots"
            callback._rows = []
            callback._task_rows = []

            combined = _combined_frame()
            for epoch in range(2):
                callback._rows.append(
                    {
                        "epoch": epoch,
                        "run_name": callback.run_name,
                        "conditioning": callback.conditioning,
                        "seed": callback.seed,
                        "train_joint_loss": 1.0 - 0.1 * epoch,
                        "val_joint_loss": 1.2 - 0.1 * epoch,
                        "train_accuracy": 0.5 + 0.1 * epoch,
                        "val_accuracy": 0.4 + 0.1 * epoch,
                        "test_joint_loss": 1.3 - 0.1 * epoch,
                        "test_primitive_loss": 1.1 - 0.1 * epoch,
                        "test_ue_fma_normalized_mse": 0.02,
                        "test_accuracy": 0.45 + 0.1 * epoch,
                        "test_macro_f1": 0.4 + 0.1 * epoch,
                        "test_weighted_f1": 0.42 + 0.1 * epoch,
                        "test_reach_f1": 0.3,
                        "test_reposition_f1": 0.35,
                    }
                )
                callback._task_rows.extend(
                    task_metric_rows(
                        combined, epoch, callback.run_name, callback.conditioning, callback.seed
                    )
                )
                callback._persist()
                written = pd.read_csv(callback.metrics_path)
                self.assertEqual(len(written), epoch + 1)

            written = pd.read_csv(callback.metrics_path)
            self.assertEqual(list(written["epoch"]), [0, 1])
            self.assertEqual(list(written.columns)[:4], list(EPOCH_METRIC_COLUMNS)[:4])
            for name in (
                "train_vs_val_loss.png",
                "train_vs_val_accuracy.png",
                "test_loss_by_epoch.png",
                "test_accuracy_by_epoch.png",
                "task_accuracy_by_epoch.png",
            ):
                self.assertTrue((callback.plot_dir / name).is_file(), name)
            self.assertTrue(callback.manifest_path.is_file())
            self.assertTrue(pd.read_csv(callback.task_metrics_path).shape[0] > 0)

            # A resumed run reloads the table and re-evaluating an epoch must not
            # duplicate its row.
            resumed = EpochMetricsCallback.__new__(EpochMetricsCallback)
            resumed.metrics_path = callback.metrics_path
            resumed.task_metrics_path = callback.task_metrics_path
            reloaded = resumed._existing_rows(callback.metrics_path)
            self.assertEqual(len(reloaded), 2)
            callback._rows.append(dict(callback._rows[-1]))
            callback._persist()
            self.assertEqual(len(pd.read_csv(callback.metrics_path)), 2)


if __name__ == "__main__":
    unittest.main()
