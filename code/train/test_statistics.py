from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from artifacts import aggregate_oof_predictions
from config import (
    CONDITIONING_NONE,
    CONDITIONING_PREFIX,
    CONDITIONING_TEXT,
    RuntimeConfig,
    VARIANTS,
)
from counting import generate_counting_artifacts
from cval_statistics import export_crossval_statistics


class StatisticsIntegrationTests(unittest.TestCase):
    def test_oof_aggregation_and_three_table_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(
                data_root=root / "data",
                output_root=root / "outputs",
                report_root=root / "reports",
                cache_root=root / "cache",
                n_folds=4,
            )
            labels = ["reach", "reposition", "rest", "stabilize", "transport"]
            assignment_rows = []
            for fold_id in range(4):
                subjects = [
                    f"s{index:02d}"
                    for index in range(fold_id * 8 + 1, fold_id * 8 + 9)
                ]
                assignment_rows.extend(
                    {
                        "subject_id": subject,
                        "fold_id": fold_id,
                        "fold_seed": 42 + fold_id,
                    }
                    for subject in subjects
                )
                for run_name, conditioning in VARIANTS:
                    # Base is the noisiest, text sits between, prefix is cleanest, so
                    # both contrasts against base have a non-zero paired delta.
                    error_period = {
                        CONDITIONING_NONE: 7,
                        CONDITIONING_TEXT: 11,
                        CONDITIONING_PREFIX: 0,
                    }[conditioning]
                    fma_error = {
                        CONDITIONING_NONE: 1.0,
                        CONDITIONING_TEXT: 0.6,
                        CONDITIONING_PREFIX: 0.2,
                    }[conditioning]
                    prediction_dir = (
                        config.output_root
                        / "folds"
                        / f"fold_{fold_id:02d}"
                        / run_name
                        / "predictions"
                    )
                    prediction_dir.mkdir(parents=True)
                    primitive_rows = []
                    combined_rows = []
                    fma_rows = []
                    for subject in subjects:
                        subject_index = int(subject[1:])
                        actual_fma = 20.0 + subject_index
                        predicted_fma = actual_fma + fma_error
                        fma_rows.append(
                            {
                                "subject_id": subject,
                                "actual_ue_fma": actual_fma,
                                "predicted_ue_fma": predicted_fma,
                                "absolute_error": abs(actual_fma - predicted_fma),
                                "run_name": run_name,
                                "fold_id": fold_id,
                                "split": "oof",
                            }
                        )
                        for horizon, actual in enumerate(labels):
                            predicted = actual
                            if error_period and (subject_index + horizon) % error_period == 0:
                                predicted = labels[(horizon + 1) % len(labels)]
                            common = {
                                "subject_id": subject,
                                "clip_id": f"{subject}_tabletop_1",
                                "task_name": "tabletop",
                                "target_frame_index": 155,
                                "horizon": horizon,
                                "actual_primitive": actual,
                                "predicted_primitive": predicted,
                                "prediction_probability": 0.8,
                                "run_name": run_name,
                                "fold_id": fold_id,
                                "split": "oof",
                            }
                            primitive_rows.append(common)
                            combined_rows.append(
                                {
                                    **common,
                                    "actual_label_id": labels.index(actual),
                                    "predicted_label_id": labels.index(predicted),
                                    "absolute_frame_index": 155 + horizon,
                                    "prediction_position": (
                                        "current" if horizon == 0 else f"future_{horizon}"
                                    ),
                                    "actual_ue_fma": actual_fma,
                                    "predicted_ue_fma": predicted_fma,
                                }
                            )
                    pd.DataFrame(primitive_rows).to_csv(
                        prediction_dir / "oof_primitive_predictions.csv", index=False
                    )
                    pd.DataFrame(fma_rows).to_csv(
                        prediction_dir / "oof_ue_fma_subject_predictions.csv", index=False
                    )
                    pd.DataFrame(combined_rows).to_csv(
                        prediction_dir / "oof_combined_window_predictions.csv", index=False
                    )

            config.output_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(assignment_rows).sort_values("subject_id").to_csv(
                config.output_root / "fold_assignments.csv", index=False
            )

            outputs = aggregate_oof_predictions(config)
            self.assertEqual(len(outputs), len(VARIANTS))
            for run_name, paths in outputs.items():
                self.assertEqual(pd.read_csv(paths["ue_fma"])["subject_id"].nunique(), 32)
                self.assertEqual(len(pd.read_csv(paths["combined"])), 160)
                generate_counting_artifacts(
                    pd.read_csv(paths["combined"]),
                    config.output_root / "oof" / run_name,
                    run_name,
                    "oof",
                )

            tables = export_crossval_statistics(config)
            self.assertEqual(set(tables), {"primitive", "ue_fma", "counting"})
            self.assertEqual(
                sorted(path.name for path in config.report_root.glob("*.csv")),
                [
                    "01_stroke_rehab_primitive_statistics.csv",
                    "02_stroke_rehab_uefma_correlation_ci.csv",
                    "03_stroke_rehab_counting_correlation_ci.csv",
                ],
            )
            primitive_table = pd.read_csv(tables["primitive"])
            # One row block per conditioning variant contrasted against base.
            conditioned_count = len(VARIANTS) - 1
            self.assertEqual(len(primitive_table), 4 * conditioned_count)
            self.assertEqual(
                primitive_table["Comparison"].nunique(), conditioned_count
            )
            self.assertTrue(
                {
                    "Comparison",
                    "Monte Carlo permutation p",
                    "Paired Cohen d",
                    "Wilcoxon p",
                    "Matched-pairs rank-biserial r",
                    "McNemar exact p",
                    "Conclusion",
                }.issubset(primitive_table.columns)
            )
            self.assertEqual(len(pd.read_csv(tables["ue_fma"])), len(VARIANTS))
            self.assertEqual(len(pd.read_csv(tables["counting"])), len(VARIANTS))


if __name__ == "__main__":
    unittest.main()
