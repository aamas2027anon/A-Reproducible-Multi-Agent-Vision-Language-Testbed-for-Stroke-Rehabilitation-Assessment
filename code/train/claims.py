from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)

from config import (
    BASE_RUN_NAME,
    CLASS_NAMES,
    CONDITIONING_NONE,
    EXPECTED_TEST_SUBJECTS,
    RuntimeConfig,
    VARIANTS,
)
from cval_statistics import DISPLAY_NAMES


KEY_COLUMNS = [
    "subject_id",
    "clip_id",
    "task_name",
    "target_frame_index",
    "horizon",
    "absolute_frame_index",
]


def export_claims_artifacts(config: RuntimeConfig) -> dict[str, Path]:
    export_dir = config.output_root / "claims_analysis_artifacts_crossval"
    export_dir.mkdir(parents=True, exist_ok=True)
    primitive_frames = {}
    reference_truth = None
    for run_name, _ in VARIANTS:
        path = (
            config.output_root
            / "full_train_test"
            / run_name
            / "predictions"
            / "test_combined_window_predictions.csv"
        )
        frame = pd.read_csv(path).sort_values(KEY_COLUMNS).reset_index(drop=True)
        if frame["subject_id"].nunique() != EXPECTED_TEST_SUBJECTS:
            raise AssertionError(f"{run_name} claims predictions do not contain eight subjects")
        if set(frame["split"]) != {"test"}:
            raise AssertionError(f"{run_name} claims rows must have split='test'")
        if set(frame["horizon"]) != set(range(5)):
            raise AssertionError(f"{run_name} claims predictions do not contain five horizons")
        if frame.duplicated(KEY_COLUMNS).any():
            raise AssertionError(f"{run_name} claims predictions contain duplicate frame keys")
        truth = frame[[*KEY_COLUMNS, "actual_primitive"]]
        if reference_truth is None:
            reference_truth = truth
        elif not truth.equals(reference_truth):
            raise AssertionError("Full-run claims ground truth is not aligned")
        primitive_frames[DISPLAY_NAMES[run_name]] = frame

    performance_rows = []
    for name, frame in primitive_frames.items():
        f1_values = precision_recall_fscore_support(
            frame["actual_primitive"],
            frame["predicted_primitive"],
            labels=list(CLASS_NAMES),
            zero_division=0,
        )[2]
        performance_rows.append(
            {
                "Comparison": name,
                **{
                    f"{label.title()} F1": value
                    for label, value in zip(CLASS_NAMES, f1_values)
                },
                "Macro F1": f1_values.mean(),
                "Accuracy": accuracy_score(
                    frame["actual_primitive"], frame["predicted_primitive"]
                ),
            }
        )
    performance = pd.DataFrame(performance_rows).set_index("Comparison")
    base_display = DISPLAY_NAMES[BASE_RUN_NAME]
    deltas = []
    for run_name, mode in VARIANTS:
        if mode == CONDITIONING_NONE:
            continue
        variant_display = DISPLAY_NAMES[run_name]
        delta = (
            (performance.loc[variant_display] - performance.loc[base_display])
            .to_frame()
            .T
        )
        delta.index = [f"Δ {mode} conditioning"]
        deltas.append(delta)
    primitive_table = pd.concat((performance, *deltas))
    primitive_table.index.name = "Model / contrast"

    uefma_frames = {}
    correlation_rows = []
    reference_uefma = None
    for run_name, _ in VARIANTS:
        path = (
            config.output_root
            / "full_train_test"
            / run_name
            / "predictions"
            / "test_ue_fma_subject_predictions.csv"
        )
        frame = pd.read_csv(path).sort_values("subject_id").reset_index(drop=True)
        if len(frame) != EXPECTED_TEST_SUBJECTS or not frame["subject_id"].is_unique:
            raise AssertionError("Claims UE-FMA predictions require eight unique subjects")
        if set(frame["split"]) != {"test"}:
            raise AssertionError("Claims UE-FMA rows must have split='test'")
        truth = frame[["subject_id", "actual_ue_fma"]]
        if reference_uefma is None:
            reference_uefma = truth
        elif not truth.equals(reference_uefma):
            raise AssertionError("Claims UE-FMA ground truth is not aligned")
        name = DISPLAY_NAMES[run_name]
        correlation = pearsonr(frame["actual_ue_fma"], frame["predicted_ue_fma"])
        uefma_frames[name] = frame
        correlation_rows.append(
            {
                "Variant": name,
                "Pearson r": correlation.statistic,
                "MAE": mean_absolute_error(
                    frame["actual_ue_fma"], frame["predicted_ue_fma"]
                ),
                "R²": r2_score(frame["actual_ue_fma"], frame["predicted_ue_fma"]),
                "Target r >= 0.65": bool(correlation.statistic >= 0.65),
            }
        )
    correlation_table = pd.DataFrame(correlation_rows)

    outputs = {
        "table1_csv": export_dir / "table1_metrics.csv",
        "table1_tex": export_dir / "table1_metrics.tex",
        "table2_csv": export_dir / "table2_uefma.csv",
        "table2_tex": export_dir / "table2_uefma.tex",
        "figure": export_dir / "uefma.png",
    }
    primitive_table.to_csv(outputs["table1_csv"], float_format="%.4f")
    primitive_table.rename(index={"Δ patient prefix": r"$\Delta$ patient prefix"}).to_latex(
        outputs["table1_tex"],
        caption="Full five-position target-sequence primitive performance.",
        label="tab:primitive-performance",
        float_format="%.3f",
        escape=False,
    )
    correlation_table.to_csv(outputs["table2_csv"], index=False, float_format="%.4f")
    correlation_table.rename(
        columns={"Pearson r": r"Pearson $r$", "R²": r"$R^2$"}
    ).to_latex(
        outputs["table2_tex"],
        index=False,
        caption="Subject-level UE-FMA clinical alignment.",
        label="tab:uefma-alignment",
        float_format="%.3f",
        escape=False,
    )

    variant_names = [DISPLAY_NAMES[run_name] for run_name, _ in VARIANTS]
    figure, axes = plt.subplots(
        1,
        len(variant_names),
        figsize=(5 * len(variant_names), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, name in zip(np.atleast_1d(axes), variant_names):
        frame = uefma_frames[name]
        actual = frame["actual_ue_fma"]
        predicted = frame["predicted_ue_fma"]
        correlation = pearsonr(actual, predicted).statistic
        regression = linregress(actual, predicted)
        axis.scatter(actual, predicted, s=52, color="#2f6b8a", edgecolor="#173b4d")
        axis.plot([0, 66], [0, 66], color="#777777", linestyle=":")
        fit_x = np.array([actual.min(), actual.max()])
        axis.plot(fit_x, regression.intercept + regression.slope * fit_x, color="#c05a2b")
        axis.set_title(f"{name}\nr={correlation:.3f}")
        axis.set_xlim(0, 66)
        axis.set_ylim(0, 66)
        axis.grid(alpha=0.25)
    figure.supxlabel("Actual UE-FMA")
    figure.supylabel("Predicted UE-FMA")
    figure.savefig(outputs["figure"], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return outputs
