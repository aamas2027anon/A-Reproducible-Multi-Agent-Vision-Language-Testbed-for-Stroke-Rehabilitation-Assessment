from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from counting import (
    add_count_errors,
    assert_count_reconciliation,
    build_count_aggregations,
    collapse_events,
    count_video_primitives,
    generate_counting_artifacts,
)


class CountingTests(unittest.TestCase):
    @staticmethod
    def _timeline() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "subject_id": ["s01"] * 8,
                "task_name": ["tabletop"] * 8,
                "clip_id": ["clip"] * 8,
                "absolute_frame_index": list(range(8)),
                "actual_label_id": [0, 0, 0, 4, 4, 4, 3, 0],
                "predicted_label_id": [0, 0, 0, 4, 4, 4, 3, 0],
                "run_name": ["base"] * 8,
                "fold_id": [0] * 8,
                "split": ["oof"] * 8,
            }
        )

    def test_run_length_counts_and_reconciliation(self) -> None:
        video = count_video_primitives(self._timeline())
        self.assertEqual(int(video.query("primitive == 'reach'")["actual_count"].iloc[0]), 2)
        self.assertEqual(int(video.query("primitive == 'transport'")["actual_count"].iloc[0]), 1)
        self.assertEqual(int(video.query("primitive == 'stabilize'")["actual_count"].iloc[0]), 1)
        self.assertEqual(int(video["actual_count"].sum()), 4)
        aggregations = build_count_aggregations(video)
        assert_count_reconciliation(aggregations)
        self.assertEqual(int(aggregations["subject_total"]["actual_count"].iloc[0]), 4)

    def test_gaps_split_events_and_zero_actual_percentage_is_nan(self) -> None:
        gap = pd.DataFrame(
            {"absolute_frame_index": [0, 1, 3], "predicted_label_id": [0, 0, 0]}
        )
        self.assertEqual(len(collapse_events(gap, "predicted_label_id", 1)), 2)
        errors = add_count_errors(
            pd.DataFrame({"actual_count": [0, 2], "predicted_count": [1, 1]})
        )
        self.assertTrue(np.isnan(errors["predicted_percent_of_actual"].iloc[0]))
        self.assertEqual(errors["predicted_percent_of_actual"].iloc[1], 50.0)

    def test_artifact_schema_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = generate_counting_artifacts(
                self._timeline(), Path(temporary), "base", "oof"
            )
            self.assertTrue(result["manifest"].is_file())
            for path in result["predictions"].values():
                self.assertTrue(path.is_file())
            subject_total = pd.read_csv(result["predictions"]["subject_total"])
            self.assertTrue(
                {
                    "actual_count",
                    "predicted_count",
                    "count_error",
                    "absolute_percent_error",
                }.issubset(subject_total.columns)
            )


if __name__ == "__main__":
    unittest.main()
