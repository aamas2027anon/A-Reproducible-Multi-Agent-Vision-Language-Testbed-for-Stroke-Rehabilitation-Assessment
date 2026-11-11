"""Multi-seed VLM measurements and conditioning comparisons from saved predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from config import (
    BASE_RUN_NAME,
    CONDITIONING_NONE,
    LABEL_VOCAB,
    NUM_PRIMITIVE_CLASSES,
    SEEDS,
    OUTPUT_DIR,
    experiment_path,
    VARIANTS,
    RuntimeConfig,
)
from multiseed import seed_run_dir, seeds_root
from sign_flip import sign_flip_pvalue

DISPLAY_NAMES = {
    "qwen_without_patient_conditioning": "Base",
    "qwen_with_patient_text": "Base + patient text",
    "qwen_with_patient_prefix": "Base + patient prefix",
}
CONDITIONED_VARIANTS = tuple(
    (run_name, mode) for run_name, mode in VARIANTS if mode != CONDITIONING_NONE
)
PRIMITIVE_METRICS = (
    ("accuracy", "Accuracy"),
    ("weighted_f1", "Weighted F1"),
    ("reach_f1", "Reach F1"),
    ("reposition_f1", "Reposition F1"),
)
UEFMA_METRICS = (
    ("ue_fma_pearson_r", "UE-FMA r"),
    ("ue_fma_mae", "UE-FMA MAE"),
    ("ue_fma_rmse", "UE-FMA RMSE"),
)
COMPARISON_TABLES = {
    "qwen_with_patient_text": "vlm_text_vs_base.csv",
    "qwen_with_patient_prefix": "vlm_prefix_vs_base.csv",
}


# --------------------------------------------------------------------------------
# Per-seed metrics
# --------------------------------------------------------------------------------


def primitive_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Frame-level metrics over every scored target (all horizons)."""
    if frame.empty:
        raise ValueError("Cannot calculate primitive metrics from an empty frame")
    actual = frame["actual_label_id"].to_numpy(dtype=int)
    predicted = frame["predicted_label_id"].to_numpy(dtype=int)
    labels = list(range(NUM_PRIMITIVE_CLASSES))
    f1 = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)[2]
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "weighted_f1": float(
            f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "reach_f1": float(f1[LABEL_VOCAB["reach"]]),
        "reposition_f1": float(f1[LABEL_VOCAB["reposition"]]),
    }


def _safe_pearson(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    if (
        len(actual_values) < 2
        or np.unique(actual_values).size < 2
        or np.unique(predicted_values).size < 2
    ):
        return float("nan")
    return float(pearsonr(actual_values, predicted_values).statistic)


def _seed_is_complete(config: RuntimeConfig, seed: int) -> bool:
    for run_name, _ in VARIANTS:
        predictions = seed_run_dir(config, seed, run_name) / "predictions"
        for name in ("test_combined_window_predictions.csv", "test_ue_fma_subject_predictions.csv"):
            if not (predictions / name).is_file():
                print(f"Seed {seed}: {run_name} has no {name}; excluding this seed")
                return False
    return True


def _seed_variant_row(config: RuntimeConfig, seed: int, run_name: str, conditioning: str,
                      combined: pd.DataFrame) -> dict[str, Any]:
    run_dir = seed_run_dir(config, seed, run_name)
    manifest_path = run_dir / "training_manifest.json"
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed")
        if recorded is not None and int(recorded) != int(seed):
            raise AssertionError(
                f"Seed {seed}: {run_name} was trained at seed {recorded}; "
                "the variants at this seed are not paired"
            )
    ue_fma = pd.read_csv(run_dir / "predictions" / "test_ue_fma_subject_predictions.csv")
    actual, predicted = ue_fma["actual_ue_fma"], ue_fma["predicted_ue_fma"]
    return {
        "seed": seed,
        "run_name": run_name,
        "conditioning": conditioning,
        "variant": DISPLAY_NAMES[run_name],
        "n_test_subjects": int(combined["subject_id"].nunique()),
        **primitive_metrics(combined),
        "ue_fma_pearson_r": _safe_pearson(actual, predicted),
        "ue_fma_mae": float(mean_absolute_error(actual, predicted)),
        "ue_fma_rmse": float(mean_squared_error(actual, predicted) ** 0.5),
    }


def collect_seed_metrics(config: RuntimeConfig, seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """One row per (seed, variant); a seed missing any variant is left out."""
    rows = []
    keys = ["subject_id", "clip_id", "absolute_frame_index"]
    for seed in seeds:
        if not _seed_is_complete(config, seed):
            continue
        reference = None
        for run_name, conditioning in VARIANTS:
            path = seed_run_dir(config, seed, run_name) / "predictions" / "test_combined_window_predictions.csv"
            combined = pd.read_csv(path)
            if combined.duplicated(keys).any():
                raise ValueError(f"Seed {seed}, {run_name}: duplicate scored frames")
            truth = combined[keys + ["actual_label_id"]].sort_values(keys).reset_index(drop=True)
            if reference is not None and not truth.equals(reference):
                raise ValueError(f"Seed {seed}: conditioning variants score different frames or labels")
            reference = truth
            rows.append(_seed_variant_row(config, seed, run_name, conditioning, combined))
    if not rows:
        raise RuntimeError(f"No seed has all variants complete under {seeds_root(config)}")
    return pd.DataFrame(rows).sort_values(["seed", "run_name"]).reset_index(drop=True)


# --------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------


def _by_seed(seed_level: pd.DataFrame, run_name: str, column: str) -> pd.Series:
    selected = seed_level.loc[seed_level["run_name"] == run_name, ["seed", column]]
    return selected.set_index("seed")[column].sort_index()


def vlm_significance(seed_level: pd.DataFrame) -> pd.DataFrame:
    """Each conditioning variant against Base, one row per metric."""
    rows = []
    for run_name, _ in CONDITIONED_VARIANTS:
        for column, label in PRIMITIVE_METRICS:
            base = _by_seed(seed_level, BASE_RUN_NAME, column)
            variant = _by_seed(seed_level, run_name, column)
            if not base.index.is_unique or not variant.index.is_unique or not base.index.equals(variant.index):
                raise ValueError("Conditioning comparisons need one matching measurement per seed")
            differences = variant - base
            rows.append(
                {
                    "Comparison": f"{DISPLAY_NAMES[run_name]} vs {DISPLAY_NAMES[BASE_RUN_NAME]}",
                    "Metric": label,
                    "n seeds": len(differences),
                    "Base mean": float(base.mean()),
                    "Base SD": float(base.std(ddof=1)),
                    "Variant mean": float(variant.mean()),
                    "Variant SD": float(variant.std(ddof=1)),
                    "Mean paired delta": float(differences.mean()),
                    "p": sign_flip_pvalue(differences.to_numpy()),
                    "Better": int((differences > 0).sum()),
                    "Worse": int((differences < 0).sum()),
                    "run_name": run_name,
                }
            )
    return pd.DataFrame(rows)


def uefma_table(seed_level: pd.DataFrame) -> pd.DataFrame:
    """Subject-level UE-FMA metrics per variant: mean and SD over seeds."""
    rows = []
    for column, label in UEFMA_METRICS:
        row: dict[str, Any] = {"Metric": label, "n seeds": int(seed_level["seed"].nunique())}
        for run_name, _ in VARIANTS:
            values = _by_seed(seed_level, run_name, column)
            row[f"{DISPLAY_NAMES[run_name]} mean"] = float(values.mean())
            row[f"{DISPLAY_NAMES[run_name]} SD"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def uefma_subject_predictions(config: RuntimeConfig, seeds: list[int]) -> pd.DataFrame:
    """Every seed's subject-level UE-FMA predictions, stacked for plotting."""
    frames = []
    for seed in seeds:
        for run_name, _ in VARIANTS:
            path = seed_run_dir(config, seed, run_name) / "predictions" / "test_ue_fma_subject_predictions.csv"
            frame = pd.read_csv(path)
            frame["seed"] = seed
            frame["variant"] = DISPLAY_NAMES[run_name]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------


def export_report(
    config: RuntimeConfig, output_dir: Path | None = None, *, seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Path]:
    seed_level = collect_seed_metrics(config, seeds)
    requested_seeds = seeds
    seeds = sorted(int(seed) for seed in seed_level["seed"].unique())
    print(f"Reporting {len(seeds)} of {len(requested_seeds)} seeds: {seeds}")
    output = Path(output_dir) if output_dir is not None else experiment_path(OUTPUT_DIR / "report", config.clip_percentage)
    if len(seeds) == 1:
        output = output / f"single_seed_{seeds[0]}"
        output.mkdir(parents=True, exist_ok=True)
        path = output / "vlm_metrics.csv"
        seed_level.to_csv(path, index=False)
        return {"vlm_metrics": path}

    import figures

    tables = output / "tables"
    plots = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    outputs = {"vlm_seed_metrics": tables / "vlm_seed_metrics.csv"}
    seed_level.to_csv(outputs["vlm_seed_metrics"], index=False)
    outputs["vlm_uefma"] = tables / "vlm_uefma.csv"
    uefma_table(seed_level).to_csv(outputs["vlm_uefma"], index=False, float_format="%.6f")
    significance = vlm_significance(seed_level)
    for run_name, filename in COMPARISON_TABLES.items():
        key = Path(filename).stem
        outputs[key] = tables / filename
        significance.loc[significance["run_name"] == run_name].drop(columns="run_name").to_csv(
            outputs[key], index=False, float_format="%.6f"
        )

    outputs["vlm_metric_spread"] = figures.vlm_metric_spread(
        seed_level, significance, plots / "vlm_metric_spread.png"
    )
    outputs["vlm_uefma_scatter"] = figures.vlm_uefma(
        uefma_subject_predictions(config, seeds),
        seed_level,
        plots / "vlm_uefma_actual_vs_predicted.png",
    )
    return outputs
