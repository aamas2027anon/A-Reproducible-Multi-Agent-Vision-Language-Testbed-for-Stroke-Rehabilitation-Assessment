from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ceiling_report import (
    FORBIDDEN_COLUMN_TOKENS,
    _assert_no_inferential_columns,
    _best_epoch,
    ceiling_analysis_dir,
    export_ceiling_report,
)
from config import CEILING_SEED, CEILING_STAGE, RuntimeConfig, VARIANTS
from cval_statistics import DISPLAY_NAMES
from multiseed import seed_run_dir
from test_multiseed import MultiseedStatisticsTests


class BestEpochTests(unittest.TestCase):
    @staticmethod
    def _epochs(run_name: str, accuracies: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epoch": range(len(accuracies)),
                "test_accuracy": accuracies,
                "variant": DISPLAY_NAMES[run_name],
            }
        )

    def test_peak_in_the_middle_is_not_still_improving(self) -> None:
        run_name = VARIANTS[0][0]
        result = _best_epoch(self._epochs(run_name, [0.1, 0.5, 0.4, 0.3]), run_name)
        self.assertEqual(result["best_epoch"], 2)  # reported 1-based
        self.assertEqual(result["epochs_trained"], 4)
        self.assertAlmostEqual(result["best_test_accuracy"], 0.5)
        self.assertAlmostEqual(result["final_test_accuracy"], 0.3)
        self.assertFalse(result["still_improving_at_last_epoch"])

    def test_peak_at_the_end_flags_still_improving(self) -> None:
        run_name = VARIANTS[0][0]
        result = _best_epoch(self._epochs(run_name, [0.1, 0.2, 0.3]), run_name)
        self.assertEqual(result["best_epoch"], 3)
        self.assertTrue(result["still_improving_at_last_epoch"])

    def test_missing_variant_is_not_fatal(self) -> None:
        result = _best_epoch(pd.DataFrame(columns=["epoch", "test_accuracy", "variant"]),
                             VARIANTS[0][0])
        self.assertIsNone(result["best_epoch"])
        self.assertEqual(result["epochs_trained"], 0)


class NoStatisticsGuardTests(unittest.TestCase):
    def test_inferential_columns_are_rejected(self) -> None:
        for column in (
            "Monte Carlo permutation p",
            "Bootstrap 95% CI low",
            "Mean paired_delta",
            "Wilcoxon p",
            "Paired Cohen d",
            "McNemar exact p",
        ):
            with self.assertRaises(AssertionError):
                _assert_no_inferential_columns(pd.DataFrame(columns=[column]), "table")

    def test_measurement_columns_pass(self) -> None:
        # "reach_precision" and "prediction_position" both contain "ci"; the token list
        # must stay specific enough not to reject real measurements.
        frame = pd.DataFrame(
            columns=[
                "variant",
                "accuracy",
                "weighted_f1",
                "reach_precision",
                "reposition_recall",
                "prediction_position",
                "count_mae",
                "ue_fma_pearson_r",
                "best_epoch",
                "still_improving_at_last_epoch",
            ]
        )
        _assert_no_inferential_columns(frame, "table")
        self.assertTrue(FORBIDDEN_COLUMN_TOKENS)


class CeilingReportTests(unittest.TestCase):
    def _build(self, root: Path) -> RuntimeConfig:
        """One seed's worth of the synthetic tree that test_multiseed already builds."""
        config = RuntimeConfig(
            data_root=root / "data",
            output_root=root / "outputs",
            report_root=root / "reports" / "stat_analysis_report_tables_ceiling",
            cache_root=root / "cache",
            stage=CEILING_STAGE,
        )
        for run_name, conditioning in VARIANTS:
            # _write_variant is a staticmethod, so no TestCase instance is needed.
            MultiseedStatisticsTests._write_variant(
                config, CEILING_SEED, run_name, conditioning
            )
            self._write_epoch_metrics(config, run_name, conditioning)
        return config

    @staticmethod
    def _write_epoch_metrics(
        config: RuntimeConfig, run_name: str, conditioning: str
    ) -> None:
        epoch_dir = seed_run_dir(config, CEILING_SEED, run_name) / "epochs"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for epoch in range(4):
            rows.append(
                {
                    "epoch": epoch,
                    "run_name": run_name,
                    "conditioning": conditioning,
                    "seed": CEILING_SEED,
                    "train_joint_loss": 1.0 - 0.1 * epoch,
                    "val_joint_loss": 1.2 - 0.05 * epoch,
                    "train_accuracy": 0.5 + 0.05 * epoch,
                    "val_accuracy": 0.45 + 0.04 * epoch,
                    "test_joint_loss": 1.3 - 0.06 * epoch,
                    "test_accuracy": 0.4 + 0.05 * epoch,
                }
            )
        pd.DataFrame(rows).to_csv(epoch_dir / "epoch_metrics.csv", index=False)
        task_rows = [
            {
                "epoch": epoch,
                "run_name": run_name,
                "conditioning": conditioning,
                "seed": CEILING_SEED,
                "task_name": "tabletop",
                "prediction_position": "current",
                "accuracy": 0.4 + 0.05 * epoch,
                "weighted_f1": 0.4 + 0.04 * epoch,
            }
            for epoch in range(4)
        ]
        pd.DataFrame(task_rows).to_csv(epoch_dir / "epoch_task_metrics.csv", index=False)

    def test_report_covers_three_variants_without_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._build(root)
            outputs = export_ceiling_report(config, (CEILING_SEED,))

            variants = pd.read_csv(outputs["variant_metrics"])
            self.assertEqual(len(variants), len(VARIANTS))
            self.assertTrue(variants["n_seeds"].eq(1).all())
            self.assertEqual(
                set(variants["variant"]),
                {DISPLAY_NAMES[run_name] for run_name, _ in VARIANTS},
            )
            # Accuracy rises monotonically in the fixture, so every variant is flagged
            # as still improving at the final epoch.
            self.assertTrue(variants["still_improving_at_last_epoch"].all())
            self.assertTrue(variants["best_epoch"].eq(4).all())
            _assert_no_inferential_columns(variants, "variant metrics")

            epochs = pd.read_csv(outputs["epoch_metrics"])
            self.assertEqual(len(epochs), 4 * len(VARIANTS))

            tasks = pd.read_csv(outputs["task_metrics"])
            self.assertFalse(tasks.empty)
            _assert_no_inferential_columns(tasks, "task metrics")

            plot_dir = Path(outputs["plots"])
            for name in (
                "test_accuracy_by_epoch.png",
                "test_loss_by_epoch.png",
                "train_vs_val_loss_by_variant.png",
                "task_accuracy_by_epoch.png",
            ):
                self.assertTrue((plot_dir / name).is_file(), name)
            self.assertEqual(plot_dir.parent, ceiling_analysis_dir(config))

    def test_multiple_seeds_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._build(root)
            with self.assertRaises(ValueError):
                export_ceiling_report(config, (CEILING_SEED, 43))


if __name__ == "__main__":
    unittest.main()
