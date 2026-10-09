from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from artifacts import classification_summary
from config import RuntimeConfig, VARIANTS, portable_path
from counting import build_count_aggregations, count_video_primitives


def _safe_correlation(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    if (
        len(actual_values) < 2
        or np.unique(actual_values).size < 2
        or np.unique(predicted_values).size < 2
    ):
        return float("nan")
    return float(np.corrcoef(actual_values, predicted_values)[0, 1])


def _prediction_metrics(run_dir: Path, split: str) -> dict[str, float | int]:
    prediction_dir = run_dir / "predictions"
    primitive = pd.read_csv(prediction_dir / f"{split}_combined_window_predictions.csv")
    overall = classification_summary(primitive)
    current = classification_summary(primitive.loc[primitive["horizon"] == 0])
    ue_fma = pd.read_csv(prediction_dir / f"{split}_ue_fma_subject_predictions.csv")
    ue_error = (
        ue_fma["predicted_ue_fma"].to_numpy(dtype=float)
        - ue_fma["actual_ue_fma"].to_numpy(dtype=float)
    )
    subject_counts = build_count_aggregations(
        count_video_primitives(primitive)
    )["subject_total"]
    count_error = (
        subject_counts["predicted_count"].to_numpy(dtype=float)
        - subject_counts["actual_count"].to_numpy(dtype=float)
    )
    return {
        "prediction_subject_count": int(primitive["subject_id"].nunique()),
        "primitive_all_accuracy": overall["accuracy"],
        "primitive_all_macro_f1": overall["macro_f1"],
        "primitive_all_weighted_f1": overall["weighted_f1"],
        "primitive_all_reach_f1": overall["reach_f1"],
        "primitive_all_reposition_f1": overall["reposition_f1"],
        "primitive_current_accuracy": current["accuracy"],
        "primitive_current_macro_f1": current["macro_f1"],
        "primitive_current_weighted_f1": current["weighted_f1"],
        "primitive_current_reach_f1": current["reach_f1"],
        "primitive_current_reposition_f1": current["reposition_f1"],
        "ue_fma_mae": float(np.mean(np.abs(ue_error))),
        "ue_fma_rmse": float(np.sqrt(np.mean(ue_error**2))),
        "ue_fma_pearson_r": _safe_correlation(
            ue_fma["actual_ue_fma"], ue_fma["predicted_ue_fma"]
        ),
        "count_mae": float(np.mean(np.abs(count_error))),
        "count_rmse": float(np.sqrt(np.mean(count_error**2))),
        "count_pearson_r": _safe_correlation(
            subject_counts["actual_count"], subject_counts["predicted_count"]
        ),
    }


def _summary_row(
    config: RuntimeConfig,
    training_scope: str,
    fold_id: int,
    run_name: str,
    run_dir: Path,
    split: str,
) -> dict[str, Any]:
    manifest_path = run_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_metrics = manifest.get("final_metrics", {})
    prediction_manifest = run_dir / "predictions" / "prediction_manifest.json"
    return {
        "training_scope": training_scope,
        "evaluation_split": split,
        "fold_id": fold_id,
        "n_folds": config.n_folds,
        "run_name": run_name,
        "conditioning": manifest.get("conditioning"),
        "use_patient_prefix": manifest["use_patient_prefix"],
        "training_complete": manifest["training_complete"],
        "config_id": manifest["config_id"],
        "seed": manifest["seed"],
        "train_subject_count": len(manifest["train_subjects"]),
        "validation_subject_count": len(manifest["validation_subjects"]),
        "prediction_subject_count_manifest": len(manifest["prediction_subjects"]),
        "train_subjects": ";".join(manifest["train_subjects"]),
        "validation_subjects": ";".join(manifest["validation_subjects"]),
        "prediction_subjects": ";".join(manifest["prediction_subjects"]),
        "max_epochs": manifest["max_epochs"],
        "completed_epochs": manifest["completed_epochs"],
        "resumed_from_checkpoint": manifest["resumed_from_checkpoint"],
        "reused_completed_checkpoint": manifest["reused_completed_checkpoint"],
        "early_stopping_enabled": manifest["early_stopping_enabled"],
        "early_stopped": manifest["early_stopped"],
        "stop_reason": manifest["stop_reason"],
        "early_stopping_monitor": manifest["early_stopping_monitor"],
        "early_stopping_patience": manifest["early_stopping_patience"],
        "early_stopping_min_delta": manifest["early_stopping_min_delta"],
        "best_val_joint_loss": manifest["early_stopping_best_score"],
        "best_val_epoch": manifest["early_stopping_best_epoch"],
        "early_stopping_stopped_after_epoch": manifest[
            "early_stopping_stopped_after_epoch"
        ],
        **{
            f"final_{name}": final_metrics.get(name)
            for name in (
                "train_joint_loss",
                "train_primitive_loss",
                "train_ue_fma_normalized_mse",
                "train_accuracy",
                "val_joint_loss",
                "val_primitive_loss",
                "val_ue_fma_normalized_mse",
                "val_accuracy",
            )
        },
        **_prediction_metrics(run_dir, split),
        "run_dir": portable_path(run_dir),
        "checkpoint": manifest["final_checkpoint"],
        "training_manifest": portable_path(manifest_path),
        "metrics_csv": manifest["history_path"],
        "prediction_manifest": portable_path(prediction_manifest),
    }


def export_all_runs_summary(config: RuntimeConfig) -> Path:
    rows: list[dict[str, Any]] = []
    for fold_id in range(config.n_folds):
        for run_name, _ in VARIANTS:
            run_dir = (
                config.output_root / "folds" / f"fold_{fold_id:02d}" / run_name
            )
            rows.append(
                _summary_row(
                    config,
                    "cross_validation",
                    fold_id,
                    run_name,
                    run_dir,
                    "oof",
                )
            )
    for run_name, _ in VARIANTS:
        run_dir = config.output_root / "full_train_test" / run_name
        rows.append(
            _summary_row(
                config,
                "full_fixed_validation",
                -1,
                run_name,
                run_dir,
                "test",
            )
        )
    result = pd.DataFrame(rows)
    keys = ["training_scope", "fold_id", "run_name"]
    expected_rows = len(VARIANTS) * (config.n_folds + 1)
    if result.duplicated(keys).any() or len(result) != expected_rows:
        raise AssertionError("All-runs summary does not contain every expected run")
    output = config.output_root / "all_runs_summary.csv"
    result.to_csv(output, index=False)
    return output
