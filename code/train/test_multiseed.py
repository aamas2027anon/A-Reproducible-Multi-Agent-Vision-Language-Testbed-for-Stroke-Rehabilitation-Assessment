from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CONDITIONING_NONE,
    CONDITIONING_PREFIX,
    CONDITIONING_TEXT,
    MULTISEED_STAGE,
    N_SEEDS,
    RuntimeConfig,
    VARIANTS,
    portable_path,
)
from artifacts import write_final_artifacts
from counting import generate_counting_artifacts
from multiseed import (
    load_or_create_resolved_seeds,
    resolve_seeds,
    seed_is_complete,
    seed_run_dir,
    subject_total_counts_path,
)
from multiseed_statistics import collect_seed_metrics, export_multiseed_statistics
from training import prediction_is_current


LABELS = ("reach", "reposition", "rest", "stabilize", "transport")
# Base is noisiest, text sits between, prefix is cleanest, so both contrasts against
# base carry a consistent non-zero paired delta across seeds.
ERROR_PERIOD = {CONDITIONING_NONE: 5, CONDITIONING_TEXT: 9, CONDITIONING_PREFIX: 0}
FMA_ERROR = {CONDITIONING_NONE: 3.0, CONDITIONING_TEXT: 2.0, CONDITIONING_PREFIX: 1.0}


class SeedResolutionTests(unittest.TestCase):
    def test_duplicates_are_repaired_not_rejected(self) -> None:
        resolved, repairs = resolve_seeds([9575, 9575, 1337])
        self.assertEqual(len(resolved), N_SEEDS)
        self.assertEqual(len(set(resolved)), N_SEEDS)
        self.assertEqual(resolved[:2], (9575, 1337))
        self.assertTrue(repairs)

    def test_resolution_is_deterministic(self) -> None:
        first, _ = resolve_seeds([1, 1, 2])
        second, _ = resolve_seeds([1, 1, 2])
        self.assertEqual(first, second)

    def test_overlong_list_is_trimmed(self) -> None:
        requested = list(range(100, 100 + N_SEEDS + 5))
        resolved, repairs = resolve_seeds(requested)
        self.assertEqual(resolved, tuple(requested[:N_SEEDS]))
        self.assertTrue(any("only" in line for line in repairs))

    def test_pool_is_pinned_on_first_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(root, root / "outputs", root, root)
            first = load_or_create_resolved_seeds(config, [7, 7, 8])
            # A different request must not silently repoint an in-flight sweep.
            second = load_or_create_resolved_seeds(config, [11, 12, 13])
            self.assertEqual(first, second)
            stored = json.loads(
                (config.output_root / "seeds" / "resolved_seeds.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(tuple(stored["resolved"]), first)


class EpochExtensionTests(unittest.TestCase):
    """A finished run must be extendable rather than skipped or silently reused."""

    def test_seed_manifest_skips_only_at_equal_or_higher_epochs(self) -> None:
        manifest = {"seed_complete": True, "seed": 42, "epochs": 10}
        self.assertTrue(seed_is_complete(manifest, 42, 10))
        self.assertTrue(seed_is_complete(manifest, 42, 3))
        # More epochs requested than were trained: must not skip.
        self.assertFalse(seed_is_complete(manifest, 42, 15))
        # A manifest from a different seed or an unfinished run never counts.
        self.assertFalse(seed_is_complete(manifest, 43, 10))
        self.assertFalse(seed_is_complete({"seed": 42, "epochs": 10}, 42, 10))
        # A manifest predating the epoch field is treated as not covering any request.
        self.assertFalse(seed_is_complete({"seed_complete": True, "seed": 42}, 42, 10))

    def test_prediction_manifest_is_invalidated_by_more_epochs(self) -> None:
        # config_id no longer contains the epoch count and last.ckpt keeps its path, so
        # completed_epochs is the only field separating a 10-epoch run's predictions
        # from a 15-epoch run's.
        checkpoint = Path("/tmp/last.ckpt")
        stored = {
            "prediction_complete": True,
            "config_id": "abc",
            "checkpoint": portable_path(checkpoint),
            "run_name": "qwen_with_patient_text",
            "fold_id": -1,
            "split": "test",
            "subjects": ["t01", "t02"],
            "completed_epochs": 10,
        }
        arguments = ("abc", checkpoint, "qwen_with_patient_text", -1, "test", ["t02", "t01"])
        self.assertTrue(prediction_is_current(stored, *arguments, 10))
        self.assertFalse(prediction_is_current(stored, *arguments, 15))
        # A manifest written before completed_epochs existed must not be trusted.
        legacy = {key: value for key, value in stored.items() if key != "completed_epochs"}
        self.assertFalse(prediction_is_current(legacy, *arguments, 10))


class MultiseedStatisticsTests(unittest.TestCase):
    @staticmethod
    def _write_variant(
        config: RuntimeConfig, seed: int, run_name: str, conditioning: str
    ) -> None:
        run_dir = seed_run_dir(config, seed, run_name)
        prediction_dir = run_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        # The seed shifts the error pattern so the per-seed metrics differ, while the
        # ground truth stays identical across variants and seeds.
        error_period = ERROR_PERIOD[conditioning]
        combined_rows = []
        fma_rows = []
        for subject_index in range(1, 9):
            subject = f"t{subject_index:02d}"
            actual_fma = 20.0 + subject_index
            predicted_fma = actual_fma + FMA_ERROR[conditioning] + (seed % 3) * 0.1
            fma_rows.append(
                {
                    "subject_id": subject,
                    "actual_ue_fma": actual_fma,
                    "predicted_ue_fma": predicted_fma,
                    "absolute_error": abs(actual_fma - predicted_fma),
                    "run_name": run_name,
                    "fold_id": -1,
                    "split": "test",
                }
            )
            # Vary the number of clips by subject so subject-level actual counts are
            # non-constant and the synthetic Pearson counting metric is defined.
            for clip_index in range(1 + subject_index % 3):
                clip_id = f"{subject}_tabletop_{clip_index}"
                for horizon, actual in enumerate(LABELS):
                    predicted = actual
                    marker = subject_index + horizon + clip_index + seed
                    if error_period and marker % error_period == 0:
                        predicted = LABELS[(horizon + 1) % len(LABELS)]
                    combined_rows.append(
                        {
                            "subject_id": subject,
                            "clip_id": clip_id,
                            "task_name": "tabletop",
                            "target_frame_index": 115,
                            "horizon": horizon,
                            "absolute_frame_index": 115 + horizon,
                            "prediction_position": (
                                "current" if horizon == 0 else f"future_{horizon}"
                            ),
                            "actual_primitive": actual,
                            "predicted_primitive": predicted,
                            "prediction_probability": 0.8,
                            "actual_label_id": LABELS.index(actual),
                            "predicted_label_id": LABELS.index(predicted),
                            "actual_ue_fma": actual_fma,
                            "predicted_ue_fma": predicted_fma,
                            "run_name": run_name,
                            "fold_id": -1,
                            "split": "test",
                        }
                    )
        combined = pd.DataFrame(combined_rows)
        combined.to_csv(
            prediction_dir / "test_combined_window_predictions.csv", index=False
        )
        pd.DataFrame(fma_rows).to_csv(
            prediction_dir / "test_ue_fma_subject_predictions.csv", index=False
        )
        (run_dir / "training_manifest.json").write_text(
            json.dumps({"seed": seed, "run_name": run_name, "conditioning": conditioning}),
            encoding="utf-8",
        )
        # Only the counting artifacts and the task table are needed downstream; the
        # confusion matrices in write_final_artifacts are not worth rendering here.
        generate_counting_artifacts(combined, run_dir / "final", run_name, "test")
        task_rows = [
            {
                "run_name": run_name,
                "split": "test",
                "task_name": "tabletop",
                "prediction_position": (
                    "current" if horizon == 0 else f"future_{horizon}"
                ),
                "accuracy": 1.0 - 0.01 * ERROR_PERIOD[conditioning] - 0.001 * horizon,
                "weighted_f1": 1.0 - 0.01 * ERROR_PERIOD[conditioning],
            }
            for horizon in range(len(LABELS))
        ]
        task_path = run_dir / "final" / "tables" / "task_metrics.csv"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(task_rows).to_csv(task_path, index=False)

    def _build_tree(self, config: RuntimeConfig) -> tuple[int, ...]:
        seeds = load_or_create_resolved_seeds(config)
        for seed in seeds:
            for run_name, conditioning in VARIANTS:
                self._write_variant(config, seed, run_name, conditioning)
        return seeds

    def test_seed_level_metrics_cover_every_seed_and_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                data_root=root / "data",
                output_root=root / "outputs",
                report_root=root / "reports" / "stat_analysis_report_tables_crossval",
                cache_root=root / "cache",
                stage=MULTISEED_STAGE,
            )
            seeds = self._build_tree(config)
            seed_metrics, mcnemar = collect_seed_metrics(config, seeds)
            self.assertEqual(len(seed_metrics), N_SEEDS * len(VARIANTS))
            self.assertEqual(seed_metrics["seed"].nunique(), N_SEEDS)
            self.assertEqual(seed_metrics["run_name"].nunique(), len(VARIANTS))
            self.assertFalse(mcnemar.empty)
            self.assertTrue(
                (config.output_root / "seeds" / "analysis" / "seed_level_metrics.csv").is_file()
            )

    def test_incomplete_seed_is_excluded_rather_than_mispaired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                data_root=root / "data",
                output_root=root / "outputs",
                report_root=root / "reports" / "stat_analysis_report_tables_crossval",
                cache_root=root / "cache",
                stage=MULTISEED_STAGE,
            )
            seeds = self._build_tree(config)
            dropped = seeds[0]
            subject_total_counts_path(
                seed_run_dir(config, dropped, VARIANTS[-1][0])
            ).unlink()
            seed_metrics, _ = collect_seed_metrics(config, seeds)
            self.assertNotIn(dropped, set(seed_metrics["seed"]))
            self.assertEqual(len(seed_metrics), (N_SEEDS - 1) * len(VARIANTS))

    def test_mismatched_seed_in_manifest_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                data_root=root / "data",
                output_root=root / "outputs",
                report_root=root / "reports" / "stat_analysis_report_tables_crossval",
                cache_root=root / "cache",
                stage=MULTISEED_STAGE,
            )
            seeds = self._build_tree(config)
            manifest = seed_run_dir(config, seeds[0], VARIANTS[0][0]) / "training_manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["seed"] = payload["seed"] + 1
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AssertionError):
                collect_seed_metrics(config, seeds)

    def test_three_tables_are_paired_by_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                data_root=root / "data",
                output_root=root / "outputs",
                report_root=root / "reports" / "stat_analysis_report_tables_crossval",
                cache_root=root / "cache",
                stage=MULTISEED_STAGE,
            )
            seeds = self._build_tree(config)
            outputs = export_multiseed_statistics(config, seeds)
            report_root = outputs["primitive"].parent
            self.assertEqual(report_root.name, "stat_analysis_report_tables_multiseed")
            self.assertEqual(
                sorted(path.name for path in report_root.glob("*.csv")),
                [
                    "01_stroke_rehab_primitive_statistics_multiseed.csv",
                    "02_stroke_rehab_uefma_correlation_ci_multiseed.csv",
                    "03_stroke_rehab_counting_correlation_ci_multiseed.csv",
                ],
            )

            conditioned_count = len(VARIANTS) - 1
            primitive = pd.read_csv(outputs["primitive"])
            self.assertEqual(len(primitive), 4 * conditioned_count)
            self.assertTrue(primitive["n seeds"].eq(N_SEEDS).all())
            self.assertEqual(primitive["Comparison"].nunique(), conditioned_count)
            self.assertTrue(
                {"Mean paired delta", "Monte Carlo permutation p", "Seeds improved"}
                .issubset(primitive.columns)
            )

            # The reported delta must equal the mean of the per-seed deltas.
            seed_metrics = pd.read_csv(outputs["seed_level_metrics"])
            base_name = VARIANTS[0][0]
            variant_name = VARIANTS[-1][0]
            base = seed_metrics.loc[
                seed_metrics["run_name"] == base_name, ["seed", "accuracy"]
            ].set_index("seed")["accuracy"]
            variant = seed_metrics.loc[
                seed_metrics["run_name"] == variant_name, ["seed", "accuracy"]
            ].set_index("seed")["accuracy"]
            expected = float((variant - base).mean())
            reported = primitive.loc[
                primitive["Metric"].eq("Accuracy")
                & primitive["Comparison"].str.contains("prefix"),
                "Mean paired delta",
            ].iloc[0]
            self.assertAlmostEqual(reported, expected, places=5)

            uefma = pd.read_csv(outputs["ue_fma"])
            self.assertEqual(
                len(uefma.loc[uefma["Row type"] == "variant"]), len(VARIANTS) * 3
            )
            self.assertEqual(
                len(uefma.loc[uefma["Row type"] == "paired delta"]),
                conditioned_count * 3,
            )
            counting = pd.read_csv(outputs["counting"])
            self.assertEqual(
                len(counting.loc[counting["Row type"] == "variant"]), len(VARIANTS) * 4
            )
            self.assertFalse(pd.read_csv(outputs["task_level"]).empty)

    def test_final_artifacts_write_task_and_counting_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                data_root=root / "data",
                output_root=root / "outputs",
                report_root=root / "reports",
                cache_root=root / "cache",
                stage=MULTISEED_STAGE,
            )
            run_name, conditioning = VARIANTS[0]
            self._write_variant(config, 11, run_name, conditioning)
            run_dir = seed_run_dir(config, 11, run_name)
            combined = pd.read_csv(
                run_dir / "predictions" / "test_combined_window_predictions.csv"
            )
            result = write_final_artifacts(combined, run_dir, run_name, "test")
            # Recorded paths are portable (never absolute); the files are under final/tables.
            tables = run_dir / "final" / "tables"
            for key in ("position_metrics", "per_class_metrics", "task_metrics"):
                self.assertFalse(Path(result[key]).is_absolute())
                self.assertTrue((tables / Path(result[key]).name).is_file())
            self.assertTrue(subject_total_counts_path(run_dir).is_file())
            task_metrics = pd.read_csv(tables / "task_metrics.csv")
            self.assertEqual(set(task_metrics["task_name"]), {"tabletop"})


if __name__ == "__main__":
    unittest.main()
