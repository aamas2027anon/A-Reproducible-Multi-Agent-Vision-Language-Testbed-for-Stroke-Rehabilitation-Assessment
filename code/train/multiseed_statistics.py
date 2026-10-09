"""Statistics over the multi-seed runs, paired by seed.

The unit of analysis is the seed: each seed contributes one test-set metric per variant,
and each conditioning variant is compared against the unconditioned base over those
paired values. The subject-level cross-validation statistics in `cval_statistics` are
unchanged and are reused here only for their test functions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from artifacts import ALIGN_COLUMNS, classification_summary
from config import (
    BASE_RUN_NAME,
    CONDITIONING_NONE,
    MULTISEED_REPORT_DIRNAME,
    N_PERMUTATIONS,
    RuntimeConfig,
    VARIANTS,
)
from cval_statistics import (
    DISPLAY_NAMES,
    METRICS,
    _mcnemar,
    _observed_conclusion,
    bootstrap_mean_ci,
    matched_pairs_rank_biserial,
    monte_carlo_sign_flip_pvalue,
    paired_cohens_d,
    wilcoxon_two_sided,
)
from multiseed import (
    load_or_create_resolved_seeds,
    seed_dir,
    seed_run_dir,
    subject_total_counts_path,
    task_metrics_path,
)


CONDITIONED_VARIANTS = tuple(
    (run_name, mode) for run_name, mode in VARIANTS if mode != CONDITIONING_NONE
)
# McNemar compares per-frame correctness, so it is undefined for weighted F1.
MCNEMAR_METRICS = ("accuracy", "reach_f1", "reposition_f1")
UEFMA_METRICS = (
    ("ue_fma_pearson_r", "UE-FMA Pearson r"),
    ("ue_fma_mae", "UE-FMA MAE"),
    ("ue_fma_rmse", "UE-FMA RMSE"),
)
COUNTING_METRICS = (
    ("count_pearson_r", "Count Pearson r"),
    ("count_mae", "Count MAE"),
    ("count_rmse", "Count RMSE"),
    ("count_mape", "Count MAPE"),
)


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


# --------------------------------------------------------------------------------
# Step 1: one row per (seed, variant)
# --------------------------------------------------------------------------------


def _load_seed_variant_frames(
    config: RuntimeConfig, seed: int
) -> dict[str, pd.DataFrame] | None:
    """Aligned per-window predictions for all variants at one seed, or None if partial."""
    frames: dict[str, pd.DataFrame] = {}
    for run_name, _ in VARIANTS:
        run_dir = seed_run_dir(config, seed, run_name)
        combined_path = (
            run_dir / "predictions" / "test_combined_window_predictions.csv"
        )
        if not combined_path.is_file() or not subject_total_counts_path(run_dir).is_file():
            print(f"Seed {seed}: {run_name} is incomplete; excluding this seed")
            return None
        manifest_path = run_dir / "training_manifest.json"
        if manifest_path.is_file():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed")
            if recorded is not None and int(recorded) != int(seed):
                raise AssertionError(
                    f"Seed {seed}: {run_name} was trained at seed {recorded}; "
                    "the variants at this seed are not paired"
                )
        frames[run_name] = (
            pd.read_csv(combined_path).sort_values(ALIGN_COLUMNS).reset_index(drop=True)
        )
    reference = None
    for run_name, frame in frames.items():
        truth = frame[[*ALIGN_COLUMNS, "actual_primitive"]]
        if reference is None:
            reference = truth
        elif not truth.equals(reference):
            raise AssertionError(
                f"Seed {seed}: {run_name} ground truth is not aligned with "
                f"{BASE_RUN_NAME}"
            )
    return frames


def _seed_variant_row(
    config: RuntimeConfig,
    seed: int,
    run_name: str,
    conditioning: str,
    combined: pd.DataFrame,
) -> dict[str, Any]:
    run_dir = seed_run_dir(config, seed, run_name)
    summary = classification_summary(combined)
    ue_fma = pd.read_csv(
        run_dir / "predictions" / "test_ue_fma_subject_predictions.csv"
    )
    counts = pd.read_csv(subject_total_counts_path(run_dir))
    valid_percent = counts["actual_count"].ne(0)
    return {
        "seed": seed,
        "run_name": run_name,
        "conditioning": conditioning,
        "variant": DISPLAY_NAMES[run_name],
        "n_test_subjects": int(combined["subject_id"].nunique()),
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "weighted_f1": summary["weighted_f1"],
        "reach_f1": summary["reach_f1"],
        "reposition_f1": summary["reposition_f1"],
        "ue_fma_pearson_r": _safe_pearson(
            ue_fma["actual_ue_fma"], ue_fma["predicted_ue_fma"]
        ),
        "ue_fma_mae": float(
            mean_absolute_error(ue_fma["actual_ue_fma"], ue_fma["predicted_ue_fma"])
        ),
        "ue_fma_rmse": float(
            mean_squared_error(ue_fma["actual_ue_fma"], ue_fma["predicted_ue_fma"]) ** 0.5
        ),
        "count_pearson_r": _safe_pearson(
            counts["actual_count"], counts["predicted_count"]
        ),
        "count_mae": float(
            mean_absolute_error(counts["actual_count"], counts["predicted_count"])
        ),
        "count_rmse": float(
            mean_squared_error(counts["actual_count"], counts["predicted_count"]) ** 0.5
        ),
        "count_mape": float(counts.loc[valid_percent, "absolute_percent_error"].mean()),
    }


def collect_seed_metrics(
    config: RuntimeConfig, seeds: Sequence[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the per-(seed, variant) metric table and the per-seed McNemar table.

    A seed missing any variant is excluded from the paired analysis rather than
    silently compared against a different seed's run.
    """
    resolved = tuple(seeds) if seeds is not None else load_or_create_resolved_seeds(config)
    rows: list[dict[str, Any]] = []
    mcnemar_rows: list[dict[str, Any]] = []
    for seed in resolved:
        frames = _load_seed_variant_frames(config, seed)
        if frames is None:
            continue
        for run_name, conditioning in VARIANTS:
            rows.append(
                _seed_variant_row(config, seed, run_name, conditioning, frames[run_name])
            )
        for run_name, _ in CONDITIONED_VARIANTS:
            for metric in MCNEMAR_METRICS:
                result = _mcnemar(frames[BASE_RUN_NAME], frames[run_name], metric)
                mcnemar_rows.append(
                    {
                        "seed": seed,
                        "comparison": f"{DISPLAY_NAMES[run_name]} vs {DISPLAY_NAMES[BASE_RUN_NAME]}",
                        "run_name": run_name,
                        "metric": metric,
                        **result,
                    }
                )
    seed_metrics = pd.DataFrame(rows)
    if seed_metrics.empty:
        raise RuntimeError(
            "No seed has all variants complete; nothing to aggregate. "
            f"Looked under {seed_dir(config, resolved[0]).parent}"
        )
    seed_metrics = seed_metrics.sort_values(["seed", "run_name"]).reset_index(drop=True)
    analysis_dir = config.output_root / "seeds" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(analysis_dir / "seed_level_metrics.csv", index=False)
    mcnemar = pd.DataFrame(mcnemar_rows)
    if not mcnemar.empty:
        mcnemar.to_csv(analysis_dir / "mcnemar_by_seed.csv", index=False)
    return seed_metrics, mcnemar


# --------------------------------------------------------------------------------
# Step 2: the three report tables
# --------------------------------------------------------------------------------


def _paired_values(
    seed_metrics: pd.DataFrame, run_name: str, column: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Base and variant values for the seeds where both are present."""
    base = (
        seed_metrics.loc[seed_metrics["run_name"] == BASE_RUN_NAME, ["seed", column]]
        .set_index("seed")[column]
    )
    variant = (
        seed_metrics.loc[seed_metrics["run_name"] == run_name, ["seed", column]]
        .set_index("seed")[column]
    )
    shared = base.index.intersection(variant.index).sort_values()
    base, variant = base.loc[shared], variant.loc[shared]
    return base, variant, variant - base


def _paired_row(
    label: str,
    comparison: str,
    base: pd.Series,
    variant: pd.Series,
    differences: pd.Series,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ci_low, ci_high = bootstrap_mean_ci(differences, rng)
    permutation_p, extreme = monte_carlo_sign_flip_pvalue(differences, rng)
    wilcoxon_statistic, wilcoxon_p = wilcoxon_two_sided(differences)
    mean_difference = float(differences.mean())
    return {
        "Comparison": comparison,
        "Metric": label,
        "n seeds": len(differences),
        "Base mean": float(base.mean()),
        "Base SD": float(base.std(ddof=1)),
        "Variant mean": float(variant.mean()),
        "Variant SD": float(variant.std(ddof=1)),
        "Base → variant": f"{base.mean():.3f} → {variant.mean():.3f}",
        "Mean paired delta": mean_difference,
        "Bootstrap 95% CI low": ci_low,
        "Bootstrap 95% CI high": ci_high,
        "Δ (bootstrap 95% CI)": f"{mean_difference:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}]",
        "Paired Cohen d": paired_cohens_d(differences),
        "Permutation method": (
            "Two-sided paired Monte Carlo sign flips over seeds with plus-one correction"
        ),
        "Monte Carlo sign flips": N_PERMUTATIONS,
        "Monte Carlo extreme count": extreme,
        "Monte Carlo permutation p": permutation_p,
        "Paired permutation p": (
            f"({extreme}+1)/({N_PERMUTATIONS}+1) = {permutation_p:.6f}"
        ),
        "Seeds improved": int((differences > 0).sum()),
        "Seeds tied": int(np.isclose(differences, 0).sum()),
        "Seeds worsened": int((differences < 0).sum()),
        "Improved": f"{int((differences > 0).sum())}/{len(differences)} seeds improved",
        "Wilcoxon statistic": wilcoxon_statistic,
        "Wilcoxon p": wilcoxon_p,
        "Matched-pairs rank-biserial r": matched_pairs_rank_biserial(differences),
        "Conclusion": _observed_conclusion(mean_difference, permutation_p),
    }


def _mcnemar_summary(mcnemar: pd.DataFrame, run_name: str, metric: str) -> str:
    if mcnemar.empty:
        return "N/A: no per-seed McNemar results"
    if metric not in MCNEMAR_METRICS:
        return "N/A: McNemar tests per-frame correctness, not weighted F1"
    selected = mcnemar.loc[
        (mcnemar["run_name"] == run_name) & (mcnemar["metric"] == metric)
    ]
    if selected.empty:
        return "N/A: no per-seed McNemar results"
    pvalues = pd.to_numeric(selected["pvalue"], errors="coerce").dropna()
    significant = int((pvalues < 0.05).sum())
    return (
        f"{significant}/{len(pvalues)} seeds p<0.05 (per-seed frame McNemar; "
        f"median p={pvalues.median():.3e}; not pooled across seeds because the same "
        "frames repeat in every seed)"
    )


def _primitive_table(
    seed_metrics: pd.DataFrame, mcnemar: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for run_name, _ in CONDITIONED_VARIANTS:
        comparison = f"{DISPLAY_NAMES[run_name]} − {DISPLAY_NAMES[BASE_RUN_NAME]}"
        for metric, label in METRICS:
            base, variant, differences = _paired_values(seed_metrics, run_name, metric)
            row = _paired_row(label, comparison, base, variant, differences, rng)
            row["McNemar seeds significant"] = _mcnemar_summary(mcnemar, run_name, metric)
            rows.append(row)
    return pd.DataFrame(rows)


def _descriptive_and_delta_table(
    seed_metrics: pd.DataFrame,
    metrics: Sequence[tuple[str, str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per-variant spread across seeds, followed by the paired deltas against base."""
    rows: list[dict[str, Any]] = []
    for run_name, _ in VARIANTS:
        values = seed_metrics.loc[seed_metrics["run_name"] == run_name]
        for column, label in metrics:
            series = pd.to_numeric(values[column], errors="coerce").dropna()
            if series.empty:
                continue
            ci_low, ci_high = bootstrap_mean_ci(series, rng)
            rows.append(
                {
                    "Row type": "variant",
                    "Variant": DISPLAY_NAMES[run_name],
                    "Metric": label,
                    "n seeds": len(series),
                    "Mean": float(series.mean()),
                    "SD": float(series.std(ddof=1)),
                    "Min": float(series.min()),
                    "Max": float(series.max()),
                    "Bootstrap 95% CI of mean": f"[{ci_low:.3f}, {ci_high:.3f}]",
                }
            )
    for run_name, _ in CONDITIONED_VARIANTS:
        comparison = f"{DISPLAY_NAMES[run_name]} − {DISPLAY_NAMES[BASE_RUN_NAME]}"
        for column, label in metrics:
            base, variant, differences = _paired_values(seed_metrics, run_name, column)
            differences = differences.dropna()
            if differences.empty:
                continue
            ci_low, ci_high = bootstrap_mean_ci(differences, rng)
            permutation_p, _ = monte_carlo_sign_flip_pvalue(differences, rng)
            rows.append(
                {
                    "Row type": "paired delta",
                    "Variant": comparison,
                    "Metric": label,
                    "n seeds": len(differences),
                    "Mean": float(differences.mean()),
                    "SD": float(differences.std(ddof=1)),
                    "Min": float(differences.min()),
                    "Max": float(differences.max()),
                    "Bootstrap 95% CI of mean": f"[{ci_low:.3f}, {ci_high:.3f}]",
                    "Paired Cohen d": paired_cohens_d(differences),
                    "Monte Carlo permutation p": permutation_p,
                    "Seeds improved": int((differences > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
# Step 3: task-level metrics across seeds
# --------------------------------------------------------------------------------


def export_task_level_metrics(
    config: RuntimeConfig, seeds: Sequence[int], rng: np.random.Generator
) -> Path:
    frames = []
    for seed in seeds:
        for run_name, conditioning in VARIANTS:
            path = task_metrics_path(seed_run_dir(config, seed, run_name))
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["seed"] = seed
            frame["conditioning"] = conditioning
            frame["variant"] = DISPLAY_NAMES[run_name]
            frames.append(frame)
    analysis_dir = config.output_root / "seeds" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output = analysis_dir / "task_level_metrics.csv"
    if not frames:
        pd.DataFrame().to_csv(output, index=False)
        return output
    combined = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (task, position), group in combined.groupby(
        ["task_name", "prediction_position"], sort=True
    ):
        for run_name, _ in VARIANTS:
            values = pd.to_numeric(
                group.loc[group["run_name"] == run_name, "accuracy"], errors="coerce"
            ).dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "Row type": "variant",
                    "task_name": task,
                    "prediction_position": position,
                    "Variant": DISPLAY_NAMES[run_name],
                    "n seeds": len(values),
                    "Accuracy mean": float(values.mean()),
                    "Accuracy SD": float(values.std(ddof=1)),
                    "Weighted F1 mean": float(
                        pd.to_numeric(
                            group.loc[group["run_name"] == run_name, "weighted_f1"],
                            errors="coerce",
                        ).mean()
                    ),
                }
            )
        base = (
            group.loc[group["run_name"] == BASE_RUN_NAME, ["seed", "accuracy"]]
            .set_index("seed")["accuracy"]
        )
        for run_name, _ in CONDITIONED_VARIANTS:
            variant = (
                group.loc[group["run_name"] == run_name, ["seed", "accuracy"]]
                .set_index("seed")["accuracy"]
            )
            shared = base.index.intersection(variant.index)
            differences = (variant.loc[shared] - base.loc[shared]).dropna()
            if differences.empty:
                continue
            permutation_p, _ = monte_carlo_sign_flip_pvalue(differences, rng)
            rows.append(
                {
                    "Row type": "paired delta",
                    "task_name": task,
                    "prediction_position": position,
                    "Variant": (
                        f"{DISPLAY_NAMES[run_name]} − {DISPLAY_NAMES[BASE_RUN_NAME]}"
                    ),
                    "n seeds": len(differences),
                    "Accuracy mean": float(differences.mean()),
                    "Accuracy SD": float(differences.std(ddof=1)),
                    "Monte Carlo permutation p": permutation_p,
                    "Seeds improved": int((differences > 0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


# --------------------------------------------------------------------------------
# Step 4: plots
# --------------------------------------------------------------------------------


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def export_multiseed_plots(config: RuntimeConfig, seed_metrics: pd.DataFrame) -> Path:
    plot_dir = config.output_root / "seeds" / "analysis" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    variant_order = [DISPLAY_NAMES[run_name] for run_name, _ in VARIANTS]

    figure, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.8), squeeze=False)
    for axis, (metric, label) in zip(axes[0], METRICS):
        grouped = [
            seed_metrics.loc[seed_metrics["variant"] == name, metric].dropna().to_numpy()
            for name in variant_order
        ]
        # Set tick labels separately: the boxplot `labels` keyword was renamed in
        # recent matplotlib, and positions are stable either way.
        axis.boxplot(grouped, showmeans=True)
        axis.set_xticks(range(1, len(variant_order) + 1), variant_order)
        for index, values in enumerate(grouped, start=1):
            jitter = np.random.default_rng(config.seed + index).normal(0, 0.04, len(values))
            axis.plot(index + jitter, values, "o", alpha=0.6, color="#2f6b8a")
        axis.set(ylabel=label, title=f"{label} across seeds")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.25, axis="y")
    _save(figure, plot_dir / "metric_spread_by_variant.png")

    figure, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.8), squeeze=False)
    for axis, (metric, label) in zip(axes[0], METRICS):
        for run_name, _ in CONDITIONED_VARIANTS:
            _, _, differences = _paired_values(seed_metrics, run_name, metric)
            axis.plot(
                range(1, len(differences) + 1),
                differences.to_numpy(),
                marker="o",
                label=DISPLAY_NAMES[run_name],
            )
        axis.axhline(0, color="#777777", linewidth=1)
        axis.set(xlabel="Seed (ordered)", ylabel=f"Δ {label}", title=f"Paired Δ {label}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    _save(figure, plot_dir / "paired_delta_by_seed.png")
    return plot_dir


def export_test_accuracy_band(config: RuntimeConfig, seeds: Sequence[int]) -> Path | None:
    """Mean ± SD test accuracy per epoch, banded across seeds."""
    frames = []
    for seed in seeds:
        for run_name, _ in VARIANTS:
            path = seed_run_dir(config, seed, run_name) / "epochs" / "epoch_metrics.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["variant"] = DISPLAY_NAMES[run_name]
            frames.append(frame)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for variant, group in combined.groupby("variant", sort=True):
        stats = group.groupby("epoch")["test_accuracy"].agg(["mean", "std"]).reset_index()
        epochs = stats["epoch"] + 1
        axis.plot(epochs, stats["mean"], marker="o", label=variant)
        axis.fill_between(
            epochs,
            stats["mean"] - stats["std"].fillna(0.0),
            stats["mean"] + stats["std"].fillna(0.0),
            alpha=0.18,
        )
    axis.set(
        xlabel="Epoch",
        ylabel="Test accuracy",
        title="Test accuracy by epoch (mean ± SD across seeds)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    path = config.output_root / "seeds" / "analysis" / "plots" / "test_accuracy_by_epoch_across_seeds.png"
    _save(figure, path)
    return path


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------


def multiseed_report_root(config: RuntimeConfig) -> Path:
    """A directory of its own, so the cross-validation tables are never overwritten."""
    if config.report_root.name == MULTISEED_REPORT_DIRNAME:
        return config.report_root
    return config.report_root.parent / MULTISEED_REPORT_DIRNAME


def export_multiseed_statistics(
    config: RuntimeConfig, seeds: Sequence[int] | None = None
) -> dict[str, Path]:
    rng = np.random.default_rng(config.seed)
    resolved = tuple(seeds) if seeds is not None else load_or_create_resolved_seeds(config)
    seed_metrics, mcnemar = collect_seed_metrics(config, resolved)
    complete_seeds = tuple(sorted(seed_metrics["seed"].unique()))
    if len(complete_seeds) < len(resolved):
        missing = sorted(set(resolved) - set(complete_seeds))
        print(
            f"Aggregating {len(complete_seeds)} of {len(resolved)} seeds; "
            f"incomplete: {missing}"
        )

    primitive = _primitive_table(seed_metrics, mcnemar, rng)
    uefma = _descriptive_and_delta_table(seed_metrics, UEFMA_METRICS, rng)
    counting = _descriptive_and_delta_table(seed_metrics, COUNTING_METRICS, rng)

    report_root = multiseed_report_root(config)
    report_root.mkdir(parents=True, exist_ok=True)
    for existing in report_root.glob("*.csv"):
        existing.unlink()
    outputs = {
        "primitive": report_root / "01_stroke_rehab_primitive_statistics_multiseed.csv",
        "ue_fma": report_root / "02_stroke_rehab_uefma_correlation_ci_multiseed.csv",
        "counting": report_root / "03_stroke_rehab_counting_correlation_ci_multiseed.csv",
    }
    primitive.to_csv(outputs["primitive"], index=False, float_format="%.6f")
    uefma.to_csv(outputs["ue_fma"], index=False, float_format="%.6f")
    counting.to_csv(outputs["counting"], index=False, float_format="%.6f")
    if sorted(path.name for path in report_root.glob("*.csv")) != sorted(
        path.name for path in outputs.values()
    ):
        raise AssertionError("Multi-seed report directory must contain exactly tables 01/02/03")

    outputs["task_level"] = export_task_level_metrics(config, complete_seeds, rng)
    export_multiseed_plots(config, seed_metrics)
    export_test_accuracy_band(config, complete_seeds)
    outputs["seed_level_metrics"] = (
        config.output_root / "seeds" / "analysis" / "seed_level_metrics.csv"
    )
    return outputs
