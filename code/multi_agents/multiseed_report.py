"""Multi-seed report: the single VLM against every MAS mode, with the seed as the unit.

    python -m multi_agents.multiseed_report --root multi_agent_outputs/multiseed_text

(`python -m multi_agents.multiseed` runs it after evaluating every seed.)

Reads every `<root>/seed_<S>/eval_<split>/` written by `multi_agents.evaluate` (one
checkpoint per seed) and writes `<root>/report/`. The first part mirrors training's
multi-seed analysis; the base is the VLM and the variants are the MAS modes. The paired estimators are
training's own (`train/cval_statistics.py`).

  tables/01_primitive_significance_across_seeds.csv   paired MAS - VLM contrasts over seeds
  tables/02_variant_spread_across_seeds.csv           mean, SD, range, bootstrap CI per mode
  tables/03_task_significance_across_seeds.csv        per task at position `current`
  tables/04_per_class_significance_across_seeds.csv   per-class F1 at position `current`
  tables/05_uefma_correlation_across_seeds.csv        UE-FMA r (Fisher z), MAE, RMSE
  tables/06_task_metrics_long.csv                     every seed x mode x task x position
  tables/07_uefma_pearson_r_by_seed.csv
  tables/15_sequence_significance_across_seeds.csv    paired MAS - VLM contrasts: edit score, AER, RCE
                                                      (frame and window level)
  plots/metric_spread_by_variant.png, metric_spread_by_variant_with_pvalues.png,
        paired_delta_by_seed.png, sequence_metrics_across_seeds.png, task_metrics_by_variant.png,
        task_metric_spread_across_seeds.png, uefma_actual_vs_predicted.png,
        uefma_pearson_r_across_seeds.png
  seed_level_metrics.csv                              one row per seed x mode
  training_agreement.csv                              VLM per seed vs training's own numbers (test split only)

The second part, `mas_only/`, is about the agents themselves: it stacks each seed's
`<mode>/mas_analysis/` (see `mas_analysis.py`) and the MAS-only measures of
`comparison.csv`.

  mas_only/tables/08_change_attribution_across_seeds.csv
  mas_only/tables/09_count_changes_across_seeds.csv
  mas_only/tables/10_class_changes_across_seeds.csv
  mas_only/tables/11_safety_and_coordination_across_seeds.csv
  mas_only/tables/12_subject_deltas_across_seeds.csv
  mas_only/tables/13_ablation_vs_mas_across_seeds.csv  each ablation minus full MAS
  mas_only/tables/14_exercise_limit_across_seeds.csv   metrics on each patient's first N exercises
  mas_only/plots/change_attribution_across_seeds.png, class_f1_delta_across_seeds.png,
                 count_error_across_seeds.png, safety_tradeoff_across_seeds.png,
                 subject_accuracy_delta_across_seeds.png, ablation_effects.png,
                 exercise_limit.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel

from multi_agents import plots
from multi_agents.settings import OUTPUT_DIR, PRIMITIVES
from multi_agents.train_bridge import classification_summary, cval_statistics, train_config

stats = cval_statistics()
plt = plots.plt

BASE = "vlm"
MODES = ("vlm", "mas", "mas_no_memory", "mas_no_planning", "mas_no_safety")
SHORT = {
    "vlm": "VLM",
    "mas": "MAS",
    "mas_no_memory": "−memory",
    "mas_no_planning": "−planning",
    "mas_no_safety": "−safety",
}
FULL = "mas"
# Each is contrasted with the full MAS: what removing one agent changes.
ABLATIONS = ("mas_no_memory", "mas_no_planning", "mas_no_safety")
ABLATION_METRICS = (
    ("accuracy", "Accuracy", True),
    ("macro_f1", "Macro F1", True),
    ("accuracy_future_primitive", "Future-primitive (t+4) accuracy", True),
    ("count_mae", "Count MAE", False),
    ("edit_score_frame", "Edit score (frame)", True),
    ("unsafe_recommendation_rate", "Unsafe recommendation rate", False),
    ("coverage_frames", "Coverage (frames released)", True),
    ("unnecessary_escalation_rate", "Unnecessary escalation rate", False),
    ("ue_fma_exercise_mae", "UE-FMA MAE (per exercise)", False),
)
# History length: (column in exercise_limit.csv, label, higher is better).
LIMIT_METRICS = (
    ("accuracy_all", "Accuracy", True),
    ("accuracy_future_primitive", "Future-primitive (t+4) accuracy", True),
    ("macro_f1_all", "Macro F1", True),
    ("count_mae", "Count MAE", False),
    ("unsafe_recommendation_rate", "Unsafe recommendation rate", False),
    ("ue_fma_subject_mae", "UE-FMA MAE (subject)", False),
)
POSITION = "current"
ALPHA = 0.05
RNG_SEED = train_config.SEED
JITTER_WIDTH = 0.05
DPI = 200
TRAINING_SEED_METRICS = (
    train_config.default_output_root() / "qwen08_multiseed_counting" / "seeds" / "analysis" / "seed_level_metrics.csv"
)

# (column, label, higher is better). The first five are training's primitive metrics.
PRIMITIVE_METRICS = (
    ("accuracy", "Accuracy", True),
    ("weighted_f1", "Weighted F1", True),
    ("macro_f1", "Macro F1", True),
    ("reach_f1", "Reach F1", True),
    ("reposition_f1", "Reposition F1", True),
)
MAS_METRICS = (
    ("accuracy_future_primitive", "Future-primitive (t+4) accuracy", True),
    ("count_mae", "Count MAE", False),
    ("count_pearson_r", "Count Pearson r", True),
    ("ue_fma_mae", "UE-FMA MAE (subject)", False),
    ("ue_fma_exercise_mae", "UE-FMA MAE (per exercise)", False),
    ("unsafe_recommendation_rate", "Unsafe recommendation rate", False),
    ("coverage_frames", "Coverage (frames released)", True),
    ("edit_score_frame", "Edit score (frame)", True),
    ("action_error_rate_frame", "Action error rate (frame)", False),
    ("relative_counting_error_frame", "Relative counting error (frame)", False),
)
# Segment-sequence metrics of comparison.csv (multi_agents/sequence_metrics.py), copied per seed.
SEQUENCE_COLUMNS = tuple(
    f"{metric}_{level}"
    for level in ("frame", "window")
    for metric in ("edit_score", "action_error_rate", "relative_counting_error")
)
# All six for table 15 and sequence_metrics_across_seeds.png.
SEQUENCE_METRICS = tuple(
    (f"{metric}_{level}", f"{label} ({level})", higher)
    for level in ("frame", "window")
    for metric, label, higher in (
        ("edit_score", "Edit score", True),
        ("action_error_rate", "Action error rate", False),
        ("relative_counting_error", "Relative counting error", False),
    )
)
TASK_METRICS = (("accuracy", "Accuracy", True), ("weighted_f1", "Weighted F1", True))
SAFETY_COLUMNS = (
    "unsafe_recommendation_rate",
    "hazard_cue_rate",
    "implausible_transition_rate",
    "coverage_frames",
    "selective_accuracy",
    "withheld_frames",
    "unnecessary_escalations",
    "unnecessary_escalation_rate",
    "unnecessary_escalations_per_1000_frames",
    "clinician_flags",
    "disagreement_windows",
    "coordination_failures_unpermitted_release",
    "coordination_failures_wrong_side",
    "coordination_failure_rate",
    "step_ms_mean",
    "step_ms_p95",
    "latency_ms_per_window_agents_total",
    "latency_ms_per_exercise_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=OUTPUT_DIR / "multiseed_text", help="Directory with seed_<S>/ runs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--training-seed-metrics", type=Path, default=TRAINING_SEED_METRICS,
                        help="Training's seed_level_metrics.csv, for the VLM agreement table")
    parser.add_argument("--training-run-name", default="qwen_with_patient_text")
    parser.add_argument("--output", type=Path, default=None, help="Default: <root>/report")
    return parser.parse_args()


# ------------------------------------------------------------------ loading


def seed_runs(root: Path, split: str) -> dict[int, Path]:
    runs = {}
    for directory in root.glob("seed_*"):
        match = re.fullmatch(r"seed_(\d+)", directory.name)
        evaluation = directory / f"eval_{split}"
        if match and (evaluation / "comparison.csv").is_file():
            runs[int(match.group(1))] = evaluation
    if not runs:
        raise SystemExit(f"No seed_<S>/eval_{split}/comparison.csv under {root}")
    return dict(sorted(runs.items()))


def present_modes(runs: dict[int, Path]) -> list[str]:
    """Modes every seed has; a paired contrast needs all of them."""
    per_seed = [set(pd.read_csv(path / "comparison.csv")["mode"]) for path in runs.values()]
    common = set.intersection(*per_seed)
    if BASE not in common:
        raise SystemExit("Every seed needs the vlm mode, the base of every contrast")
    return [mode for mode in MODES if mode in common]


def _safe_pearson(actual: pd.Series, predicted: pd.Series) -> float:
    a, p = actual.to_numpy(float), predicted.to_numpy(float)
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(p).size < 2:
        return float("nan")
    return float(pearsonr(a, p).statistic)


def seed_level_row(seed: int, mode: str, evaluation: Path, split: str, comparison: pd.Series) -> dict:
    """Training's seed-level row (same definitions), plus the MAS measures of comparison.csv."""
    mode_dir = evaluation / mode
    combined = pd.read_csv(mode_dir / "predictions" / f"{split}_combined_window_predictions.csv")
    summary = classification_summary(combined)
    ue_fma = pd.read_csv(mode_dir / "predictions" / f"{split}_ue_fma_subject_predictions.csv")
    counts = pd.read_csv(mode_dir / "final" / "counting" / "predictions" / f"{split}_subject_total_counts.csv")
    ue_error = ue_fma["predicted_ue_fma"] - ue_fma["actual_ue_fma"]
    count_error = counts["predicted_count"] - counts["actual_count"]
    valid_percent = counts["actual_count"].ne(0)
    row = {
        "seed": seed,
        "run_name": mode,
        "variant": plots.MODE_LABELS[mode],
        "n_test_subjects": int(combined["subject_id"].nunique()),
        **{key: summary[key] for key in ("accuracy", "macro_f1", "weighted_f1", "reach_f1", "reposition_f1")},
        "ue_fma_pearson_r": _safe_pearson(ue_fma["actual_ue_fma"], ue_fma["predicted_ue_fma"]),
        "ue_fma_mae": float(ue_error.abs().mean()),
        "ue_fma_rmse": float(np.sqrt((ue_error**2).mean())),
        "count_pearson_r": _safe_pearson(counts["actual_count"], counts["predicted_count"]),
        "count_mae": float(count_error.abs().mean()),
        "count_rmse": float(np.sqrt((count_error**2).mean())),
        "count_mape": float(counts.loc[valid_percent, "absolute_percent_error"].mean()),
    }
    for column in ("accuracy_future_primitive", "ue_fma_exercise_mae", *SEQUENCE_COLUMNS, *SAFETY_COLUMNS):
        if column in comparison.index:
            row[column] = comparison[column]
    # Uncalibrated subject MAE (runs before the UE-FMA calibration lack it).
    if "ue_fma_subject_mae_raw" in comparison.index:
        row["ue_fma_mae_raw"] = comparison["ue_fma_subject_mae_raw"]
    return row


def load_stacked(runs: dict[int, Path], modes: list[str], relative: str) -> pd.DataFrame:
    frames = []
    for seed, evaluation in runs.items():
        for mode in modes:
            path = evaluation / mode / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}")
            frame = pd.read_csv(path)
            frame["seed"], frame["run_name"], frame["variant"] = seed, mode, plots.MODE_LABELS[mode]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ statistics


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment; NaNs pass through untouched."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (finite.size - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def add_holm(frame: pd.DataFrame, group_columns, source="Permutation p", target="Holm p (permutation)") -> pd.DataFrame:
    frame = frame.copy()
    frame[target] = np.nan
    for _, subframe in frame.groupby(list(group_columns), sort=False):
        frame.loc[subframe.index, target] = holm(subframe[source])
    frame[f"{target} < alpha"] = frame[target] < ALPHA
    return frame


def by_seed(frame: pd.DataFrame, run_name: str, column: str) -> pd.Series:
    selected = frame.loc[frame["run_name"] == run_name, ["seed", column]]
    if selected["seed"].duplicated().any():
        raise AssertionError(f"{run_name}: more than one {column} per seed")
    return pd.to_numeric(selected.set_index("seed")[column], errors="coerce").sort_index()


def paired_stats(base: pd.Series, variant: pd.Series, rng: np.random.Generator, higher_is_better: bool = True) -> dict:
    """The notebook's paired battery; `Improved` respects the metric's direction."""
    shared = base.index.intersection(variant.index).sort_values()
    differences = (variant.loc[shared] - base.loc[shared]).dropna()
    base, variant = base.loc[differences.index], variant.loc[differences.index]
    if differences.empty:
        return {"n seeds": 0}
    ci_low, ci_high = stats.bootstrap_mean_ci(differences, rng)
    permutation_p, extreme = stats.monte_carlo_sign_flip_pvalue(differences, rng)
    wilcoxon_statistic, wilcoxon_p = stats.wilcoxon_two_sided(differences)
    if np.allclose(differences, 0) or len(differences) < 2:
        t_statistic, t_p = np.nan, np.nan
    else:
        result = ttest_rel(variant.to_numpy(float), base.to_numpy(float))
        t_statistic, t_p = float(result.statistic), float(result.pvalue)
    mean_difference = float(differences.mean())
    better = differences > 0 if higher_is_better else differences < 0
    worse = differences < 0 if higher_is_better else differences > 0
    tied = np.isclose(differences, 0)
    if not np.isfinite(permutation_p) or permutation_p >= ALPHA:
        conclusion = "No significant paired difference across seeds (Monte Carlo paired p >= alpha)."
    else:
        direction = "improvement" if (mean_difference > 0) == higher_is_better else "decrease"
        conclusion = f"Significant {direction} across seeds (Monte Carlo paired p < alpha)."
    n_permutations = train_config.N_PERMUTATIONS
    return {
        "n seeds": int(len(differences)),
        "Base mean": float(base.mean()),
        "Base SD": float(base.std(ddof=1)),
        "Variant mean": float(variant.mean()),
        "Variant SD": float(variant.std(ddof=1)),
        "Base -> variant": f"{base.mean():.4f} -> {variant.mean():.4f}",
        "Mean paired delta": mean_difference,
        "Bootstrap 95% CI low": ci_low,
        "Bootstrap 95% CI high": ci_high,
        "Delta (bootstrap 95% CI)": f"{mean_difference:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]",
        "Paired Cohen d": stats.paired_cohens_d(differences),
        "Monte Carlo sign flips": n_permutations,
        "Monte Carlo extreme count": extreme,
        "Permutation p": permutation_p,
        "Permutation p (detail)": f"({extreme}+1)/({n_permutations}+1) = {permutation_p:.6f}",
        "Wilcoxon statistic": wilcoxon_statistic,
        "Wilcoxon p": wilcoxon_p,
        "Matched-pairs rank-biserial r": stats.matched_pairs_rank_biserial(differences),
        "Paired t statistic": t_statistic,
        "Paired t p": t_p,
        "Higher is better": higher_is_better,
        "Seeds improved": int((better & ~tied).sum()),
        "Seeds tied": int(tied.sum()),
        "Seeds worsened": int((worse & ~tied).sum()),
        "Improved": f"{int((better & ~tied).sum())}/{len(differences)} seeds improved",
        "Conclusion": conclusion,
    }


def fisher_z(values) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(values, dtype=float), -0.999999, 0.999999))


def spread_row(series: pd.Series, rng: np.random.Generator) -> dict:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {"n seeds": 0}
    ci_low, ci_high = stats.bootstrap_mean_ci(series, rng)
    return {
        "n seeds": int(len(series)),
        "Mean": float(series.mean()),
        "SD": float(series.std(ddof=1)),
        "Min": float(series.min()),
        "Max": float(series.max()),
        "Bootstrap 95% CI of mean": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "Bootstrap CI low": ci_low,
        "Bootstrap CI high": ci_high,
    }


def comparison_name(mode: str) -> str:
    return f"{plots.MODE_LABELS[mode]} - {plots.MODE_LABELS[BASE]}"


# ------------------------------------------------------------------ report tables


def primitive_table(seed_level: pd.DataFrame, variants: list[str], rng, metrics=None) -> pd.DataFrame:
    rows = []
    for mode in variants:
        for column, label, higher in metrics or (*PRIMITIVE_METRICS, *MAS_METRICS):
            if column not in seed_level:
                continue
            rows.append(
                {
                    "Comparison": comparison_name(mode),
                    "Metric": label,
                    "metric_column": column,
                    **paired_stats(by_seed(seed_level, BASE, column), by_seed(seed_level, mode, column), rng, higher),
                }
            )
    return add_holm(pd.DataFrame(rows), ["Comparison"])


def spread_table(seed_level: pd.DataFrame, modes: list[str], rng) -> pd.DataFrame:
    rows = []
    for mode in modes:
        values = seed_level.loc[seed_level["run_name"] == mode]
        for column, label, _ in (*PRIMITIVE_METRICS, *MAS_METRICS):
            if column in values:
                rows.append({"Variant": plots.MODE_LABELS[mode], "Metric": label, **spread_row(values[column], rng)})
    return pd.DataFrame(rows)


def task_table(task_current: pd.DataFrame, tasks: list[str], modes: list[str], variants: list[str], rng) -> pd.DataFrame:
    variant_rows, delta_rows = [], []
    for task in tasks:
        group = task_current.loc[task_current["task_name"] == task]
        for column, label, higher in TASK_METRICS:
            for mode in modes:
                series = by_seed(group, mode, column).dropna()
                variant_rows.append(
                    {
                        "Row type": "variant",
                        "task_name": task,
                        "prediction_position": POSITION,
                        "Metric": label,
                        "Variant": plots.MODE_LABELS[mode],
                        "n seeds": int(len(series)),
                        "Mean": float(series.mean()),
                        "SD": float(series.std(ddof=1)),
                        "Min": float(series.min()),
                        "Max": float(series.max()),
                    }
                )
            for mode in variants:
                delta_rows.append(
                    {
                        "Row type": "paired delta",
                        "task_name": task,
                        "prediction_position": POSITION,
                        "Metric": label,
                        "Comparison": comparison_name(mode),
                        **paired_stats(by_seed(group, BASE, column), by_seed(group, mode, column), rng, higher),
                    }
                )
    deltas = add_holm(pd.DataFrame(delta_rows), ["Comparison", "Metric"])
    return pd.concat([pd.DataFrame(variant_rows), deltas], ignore_index=True)


def per_class_table(per_class_current: pd.DataFrame, classes: list[str], modes: list[str], variants: list[str], rng) -> pd.DataFrame:
    variant_rows, delta_rows = [], []
    for label in classes:
        group = per_class_current.loc[per_class_current["label"] == label]
        for mode in modes:
            series = by_seed(group, mode, "f1").dropna()
            variant_rows.append(
                {
                    "Row type": "variant",
                    "Class": label,
                    "prediction_position": POSITION,
                    "Variant": plots.MODE_LABELS[mode],
                    "n seeds": int(len(series)),
                    "F1 mean": float(series.mean()),
                    "F1 SD": float(series.std(ddof=1)),
                    "Support": float(by_seed(group, mode, "support").mean()),
                }
            )
        for mode in variants:
            delta_rows.append(
                {
                    "Row type": "paired delta",
                    "Class": label,
                    "prediction_position": POSITION,
                    "Metric": "F1",
                    "Comparison": comparison_name(mode),
                    **paired_stats(by_seed(group, BASE, "f1"), by_seed(group, mode, "f1"), rng),
                }
            )
    deltas = add_holm(pd.DataFrame(delta_rows), ["Comparison"])
    return pd.concat([pd.DataFrame(variant_rows), deltas], ignore_index=True)


def uefma_by_seed(uefma_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, run_name, name), group in uefma_long.groupby(["seed", "run_name", "variant"], sort=True):
        correlation = pearsonr(group["actual_ue_fma"], group["predicted_ue_fma"])
        residuals = group["actual_ue_fma"] - group["predicted_ue_fma"]
        rows.append(
            {
                "seed": seed,
                "run_name": run_name,
                "variant": name,
                "n_subjects": int(group["subject_id"].nunique()),
                "pearson_r": float(correlation.statistic),
                "pearson_p": float(correlation.pvalue),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt((residuals**2).mean())),
            }
        )
    frame = pd.DataFrame(rows)
    frame["fisher_z"] = fisher_z(frame["pearson_r"])
    return frame


def uefma_table(uefma_r: pd.DataFrame, modes: list[str], variants: list[str], rng) -> pd.DataFrame:
    variant_rows = []
    for mode in modes:
        values = uefma_r.loc[uefma_r["run_name"] == mode]
        for column, label in (("pearson_r", "Pearson r"), ("mae", "MAE"), ("rmse", "RMSE")):
            row = spread_row(values[column], rng)
            row.pop("Bootstrap CI low", None), row.pop("Bootstrap CI high", None)
            variant_rows.append(
                {
                    "Row type": "variant",
                    "Variant": plots.MODE_LABELS[mode],
                    "Metric": label,
                    **row,
                    "Seeds with r p<alpha": int((values["pearson_p"] < ALPHA).sum()) if label == "Pearson r" else pd.NA,
                }
            )
    delta_rows = []
    for mode in variants:
        for column, label, higher in (
            ("fisher_z", "Fisher z of Pearson r", True),
            ("pearson_r", "Pearson r (untransformed)", True),
            ("mae", "MAE", False),
            ("rmse", "RMSE", False),
        ):
            statistics = paired_stats(by_seed(uefma_r, BASE, column), by_seed(uefma_r, mode, column), rng, higher)
            if column == "fisher_z" and statistics.get("n seeds"):
                statistics["Delta back-transformed (tanh of mean z delta)"] = float(np.tanh(statistics["Mean paired delta"]))
            delta_rows.append({"Row type": "paired delta", "Comparison": comparison_name(mode), "Metric": label, **statistics})
    deltas = add_holm(pd.DataFrame(delta_rows), ["Comparison"])
    return pd.concat([pd.DataFrame(variant_rows), deltas], ignore_index=True)


def training_agreement(seed_level: pd.DataFrame, runs: dict[int, Path], path: Path, run_name: str) -> pd.DataFrame:
    """The VLM mode per seed next to training's own seed-level numbers for the same checkpoint."""
    vlm = seed_level.loc[seed_level["run_name"] == BASE].set_index("seed")
    # Training's UE-FMA is uncalibrated, so compare it with the VLM's raw-scale MAE when available.
    if "ue_fma_mae_raw" in vlm and vlm["ue_fma_mae_raw"].notna().all():
        vlm = vlm.assign(ue_fma_mae=vlm["ue_fma_mae_raw"])
    columns = ["accuracy", "macro_f1", "weighted_f1", "reach_f1", "reposition_f1", "ue_fma_mae", "count_mae"]
    frame = vlm[columns].add_prefix("mas_vlm_")
    if path.is_file():
        training = pd.read_csv(path)
        training = training.loc[training["run_name"] == run_name].set_index("seed")
        frame = frame.join(training[columns].add_prefix("training_"), how="left")
        for column in columns:
            frame[f"delta_{column}"] = frame[f"mas_vlm_{column}"] - frame[f"training_{column}"]
    for seed, evaluation in runs.items():
        reference = evaluation / "reference_agreement.json"
        if reference.is_file():
            frame.loc[seed, "reference_prediction_agreement"] = json.loads(reference.read_text())["agreement"]
        manifest = evaluation / "run_manifest.json"
        if manifest.is_file():
            frame.loc[seed, "checkpoint"] = json.loads(manifest.read_text())["checkpoint"]
    return frame.reset_index()


# ------------------------------------------------------------------ report plots


def strip_box(axis, groups, labels, colors=None, rng_seed=RNG_SEED):
    """Box plot with every seed jittered on top (the notebook's idiom)."""
    axis.boxplot(groups, showmeans=True, widths=0.55)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    for index, values in enumerate(groups, start=1):
        jitter = np.random.default_rng(rng_seed + index).normal(0, JITTER_WIDTH, len(values))
        color = plots.CATEGORICAL[0] if colors is None else colors[index - 1]
        axis.plot(index + jitter, values, "o", alpha=0.65, markersize=5, color=color)
    axis.grid(alpha=0.25, axis="y")
    return axis


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def plot_metric_spread(
    seed_level, modes, metrics, path: Path, primitive_stats: pd.DataFrame | None = None, ncols: int | None = None
) -> None:
    metrics = [metric for metric in metrics if metric[0] in seed_level]  # older runs lack newer columns
    if not metrics:
        return
    n_seeds = seed_level["seed"].nunique()
    ncols = ncols or len(metrics)
    nrows = int(np.ceil(len(metrics) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 5.0 * nrows), squeeze=False)
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    for axis, (column, label, _) in zip(axes.flat, metrics):
        groups = [seed_level.loc[seed_level["run_name"] == mode, column].dropna().to_numpy() for mode in modes]
        strip_box(axis, groups, [SHORT[m] for m in modes], [plots.mode_color(m) for m in modes])
        axis.set(ylabel=label, title=f"{label} across {n_seeds} seeds")
        if primitive_stats is not None:
            lines = []
            for mode in modes[1:]:
                row = primitive_stats.loc[
                    (primitive_stats["Comparison"] == comparison_name(mode)) & (primitive_stats["metric_column"] == column)
                ]
                if len(row):
                    row = row.iloc[0]
                    lines.append(f"{SHORT[mode]}: Δ={row['Mean paired delta']:+.4f}, p={row['Permutation p']:.3g}, Holm={row['Holm p (permutation)']:.3g}")
            bottom, top = axis.get_ylim()
            axis.set_ylim(bottom - 0.30 * (top - bottom), top)
            axis.annotate("\n".join(lines), xy=(0.02, 0.02), xycoords="axes fraction", va="bottom", fontsize=7.5,
                          bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
    suffix = "; paired sign-flip p vs VLM" if primitive_stats is not None else ""
    figure.suptitle(f"Metrics across seeds (box = quartiles, triangle = mean, dots = seeds{suffix})", y=1.03)
    figure.tight_layout()
    _save(figure, path)


def plot_paired_deltas(seed_level, variants, metrics, path: Path) -> None:
    metrics = [metric for metric in metrics if metric[0] in seed_level]  # older runs lack newer columns
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.6), squeeze=False)
    for axis, (column, label, _) in zip(axes[0], metrics):
        for mode in variants:
            base, variant = by_seed(seed_level, BASE, column), by_seed(seed_level, mode, column)
            differences = (variant - base).dropna()
            axis.plot([str(s) for s in differences.index], differences.to_numpy(), marker="o",
                      color=plots.mode_color(mode), label=SHORT[mode])
        axis.axhline(0, color="#777777", linewidth=1)
        axis.set(xlabel="Seed", ylabel=f"Δ {label} (MAS − VLM)", title=f"Paired Δ {label}")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, path)


def plot_task_bars(task_current, tasks, modes, n_seeds, path: Path) -> None:
    columns = [column for column, _, _ in TASK_METRICS]
    task_means = task_current.groupby(["task_name", "run_name"])[columns].agg(["mean", "std"])
    figure, axes = plt.subplots(len(TASK_METRICS), 1, figsize=(16.0, 9.0), squeeze=False, sharex=True)
    positions = np.arange(len(tasks))
    width = 0.8 / len(modes)
    for axis, (column, label, _) in zip(axes[:, 0], TASK_METRICS):
        ceiling = 0.0
        for index, mode in enumerate(modes):
            means = [task_means.loc[(task, mode), (column, "mean")] for task in tasks]
            errors = [task_means.loc[(task, mode), (column, "std")] for task in tasks]
            offsets = positions + (index - (len(modes) - 1) / 2) * width
            axis.bar(offsets, means, width=width, yerr=errors, capsize=3, label=plots.MODE_LABELS[mode],
                     color=plots.mode_color(mode), edgecolor="white")
            ceiling = max(ceiling, max(m + (e if np.isfinite(e) else 0) for m, e in zip(means, errors)))
            for x, mean in zip(offsets, means):
                axis.text(x, mean + 0.012, f"{mean:.3f}", rotation=90, ha="center", va="bottom", fontsize=7.5, color="#333333")
        axis.set_ylim(0, ceiling * 1.22)
        axis.set(ylabel=label, title=f"{label} by task at position '{POSITION}'")
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=9, ncol=len(modes))
    axes[-1, 0].set_xticks(positions, tasks, rotation=20)
    axes[-1, 0].set_xlabel("Task (ordered by test support, descending)")
    figure.suptitle(f"Task-wise metrics, mean +- SD across {n_seeds} seeds", y=0.995)
    figure.tight_layout()
    _save(figure, path)


def plot_task_spread(task_current, tasks, modes, n_seeds, path: Path) -> None:
    figure, axes = plt.subplots(len(TASK_METRICS), len(tasks), figsize=(3.1 * len(tasks), 4.2 * len(TASK_METRICS)), squeeze=False)
    for row, (column, label, _) in enumerate(TASK_METRICS):
        for col, task in enumerate(tasks):
            axis = axes[row, col]
            group = task_current.loc[task_current["task_name"] == task]
            groups = [group.loc[group["run_name"] == mode, column].dropna().to_numpy() for mode in modes]
            strip_box(axis, groups, [SHORT[m] for m in modes], [plots.mode_color(m) for m in modes], RNG_SEED + 17 * col)
            if row == 0:
                axis.set_title(task, fontsize=11)
            if col == 0:
                axis.set_ylabel(f"{label}\n(across {n_seeds} seeds)")
            axis.tick_params(axis="x", labelsize=8, rotation=30)
    figure.suptitle(f"Task-wise metric spread across seeds at position '{POSITION}' (box = quartiles, triangle = mean, dots = seeds)", y=1.0)
    figure.tight_layout()
    _save(figure, path)


def plot_uefma_scatter(uefma_long, uefma_r, modes, n_seeds, path: Path) -> None:
    figure, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 5.2), squeeze=False, sharex=True, sharey=True)
    limits = (
        min(uefma_long["actual_ue_fma"].min(), uefma_long["predicted_ue_fma"].min()) - 3,
        max(uefma_long["actual_ue_fma"].max(), uefma_long["predicted_ue_fma"].max()) + 3,
    )
    for axis, mode in zip(axes[0], modes):
        frame = uefma_long.loc[uefma_long["run_name"] == mode]
        subjects = frame.groupby("subject_id").agg(
            actual=("actual_ue_fma", "first"), predicted_mean=("predicted_ue_fma", "mean"), predicted_sd=("predicted_ue_fma", "std")
        ).sort_values("actual")
        color = plots.mode_color(mode)
        axis.errorbar(subjects["actual"], subjects["predicted_mean"], yerr=subjects["predicted_sd"], fmt="o", markersize=7,
                      capsize=3, color=color, ecolor=color, alpha=0.9, label="Subject (mean +- SD across seeds)")
        axis.plot(limits, limits, "--", color="#777777", linewidth=1, label="Identity")
        slope, intercept = np.polyfit(subjects["actual"], subjects["predicted_mean"], 1)
        grid = np.linspace(*limits, 50)
        axis.plot(grid, slope * grid + intercept, "-", color="#b02a37", linewidth=1.4, label="Least-squares fit")
        per_seed = uefma_r.loc[uefma_r["run_name"] == mode, "pearson_r"]
        pooled = pearsonr(subjects["actual"], subjects["predicted_mean"])
        axis.annotate(
            f"per-seed r = {per_seed.mean():.3f} +- {per_seed.std(ddof=1):.3f}\n"
            f"seed-averaged r = {pooled.statistic:.3f} (p = {pooled.pvalue:.3f})\n"
            f"n = {len(subjects)} subjects, {n_seeds} seeds",
            xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        axis.set(xlim=limits, ylim=limits, xlabel="Actual UE-FMA", title=plots.MODE_LABELS[mode])
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Predicted UE-FMA")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    figure.suptitle("UE-FMA: actual vs predicted, averaged across seeds", y=1.02)
    figure.tight_layout()
    _save(figure, path)


def plot_uefma_r(uefma_r, uefma_stats, modes, variants, n_subjects, n_seeds, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    groups = [uefma_r.loc[uefma_r["run_name"] == mode, "pearson_r"].to_numpy() for mode in modes]
    strip_box(axis, groups, [plots.MODE_LABELS[m] for m in modes], [plots.mode_color(m) for m in modes])
    axis.set(ylabel=f"Pearson r ({n_subjects} held-out subjects, one r per seed)", title=f"UE-FMA correlation across {n_seeds} seeds")
    axis.tick_params(axis="x", rotation=12)
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom - 0.32 * (top - bottom), top)
    annotations = []
    for mode in variants:
        row = uefma_stats.loc[(uefma_stats["Comparison"] == comparison_name(mode)) & (uefma_stats["Metric"] == "Fisher z of Pearson r")]
        if len(row) and row.iloc[0].get("n seeds"):
            row = row.iloc[0]
            annotations.append(
                f"{SHORT[mode]} - VLM: dz = {row['Mean paired delta']:+.3f}, perm p = {row['Permutation p']:.4f}, "
                f"Holm p = {row['Holm p (permutation)']:.4f}, {row['Improved']}"
            )
    axis.annotate("Paired contrasts on Fisher z\n" + "\n".join(annotations), xy=(0.02, 0.03), xycoords="axes fraction",
                  va="bottom", fontsize=8.5, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
    figure.tight_layout()
    _save(figure, path)


# ------------------------------------------------------------------ MAS-only section


def _describe(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    """Mean, SD, min and max across seeds of each column, per group."""
    grouped = frame.groupby(keys, sort=False)[columns].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()


def mas_only(runs: dict[int, Path], variants: list[str], seed_level: pd.DataFrame, output: Path, rng) -> None:
    tables_dir, plots_dir = output / "tables", output / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    attribution = load_stacked(runs, variants, "mas_analysis/change_attribution.csv")
    counts = load_stacked(runs, variants, "mas_analysis/count_changes_by_subject.csv")
    classes = load_stacked(runs, variants, "mas_analysis/class_changes.csv")
    subjects = load_stacked(runs, variants, "mas_analysis/subject_deltas.csv")

    # 08: where the agents changed the cue, and whether it helped
    table = _describe(attribution, ["run_name", "stage"], ["frames_changed", "share_of_frames", "accuracy_before", "accuracy_after", "net_correct_frames"])
    helped = attribution.groupby(["run_name", "stage"], sort=False)["net_correct_frames"].agg(
        seeds_positive=lambda s: int((s > 0).sum()), seeds_negative=lambda s: int((s < 0).sum())
    ).reset_index()
    table.merge(helped, on=["run_name", "stage"]).to_csv(tables_dir / "08_change_attribution_across_seeds.csv", index=False, float_format="%.6f")

    # 09: counts, per primitive (and total): mean per-subject absolute error, VLM vs MAS
    per_seed = counts.groupby(["run_name", "seed", "primitive"], as_index=False).agg(
        vlm_count_mae=("vlm_abs_error", "mean"),
        mas_count_mae=("mas_abs_error", "mean"),
        mean_abs_error_change=("abs_error_change", "mean"),
        mean_count_change=("mas_minus_vlm_count", "mean"),
        subjects_improved=("effect", lambda s: int((s == "improved").sum())),
        subjects_worse=("effect", lambda s: int((s == "worse").sum())),
    )
    table = _describe(per_seed, ["run_name", "primitive"], ["vlm_count_mae", "mas_count_mae", "mean_abs_error_change", "mean_count_change", "subjects_improved", "subjects_worse"])
    table.to_csv(tables_dir / "09_count_changes_across_seeds.csv", index=False, float_format="%.6f")

    # 10: per class, all positions
    delta_columns = ["delta_precision", "delta_recall", "delta_f1", "delta_predicted_share"]
    table = _describe(classes, ["run_name", "label"], ["vlm_f1", "mas_f1", *delta_columns])
    table.to_csv(tables_dir / "10_class_changes_across_seeds.csv", index=False, float_format="%.6f")

    # 11: safety, coordination and latency; paired against the VLM where the VLM has a value
    rows = []
    for mode in variants:
        for column in SAFETY_COLUMNS:
            if column not in seed_level:
                continue
            row = {"Variant": plots.MODE_LABELS[mode], "Metric": column, **spread_row(by_seed(seed_level, mode, column), rng)}
            base = by_seed(seed_level, BASE, column)
            if base.notna().all() and not np.allclose(base.fillna(0), 0):
                higher = column in ("coverage_frames", "selective_accuracy")
                paired = paired_stats(base, by_seed(seed_level, mode, column), rng, higher)
                row.update({f"vs VLM: {key}": paired[key] for key in ("Base -> variant", "Delta (bootstrap 95% CI)", "Permutation p", "Improved")})
            rows.append(row)
    pd.DataFrame(rows).to_csv(tables_dir / "11_safety_and_coordination_across_seeds.csv", index=False, float_format="%.6f")

    # 12: per subject across seeds
    table = _describe(subjects, ["run_name", "subject_id"], ["delta_accuracy", "delta_macro_f1", "delta_unsafe_rate"])
    table.to_csv(tables_dir / "12_subject_deltas_across_seeds.csv", index=False, float_format="%.6f")

    n_seeds = len(runs)
    colors = [plots.mode_color(m) for m in variants]

    # change attribution: one box per mode, one panel per stage
    stages = list(dict.fromkeys(attribution["stage"]))
    figure, axes = plt.subplots(1, len(stages), figsize=(3.0 * len(stages), 4.6), squeeze=False)
    for axis, stage in zip(axes[0], stages):
        selected = attribution.loc[attribution["stage"] == stage]
        groups = [selected.loc[selected["run_name"] == m, "net_correct_frames"].to_numpy() for m in variants]
        strip_box(axis, groups, [SHORT[m] for m in variants], colors)
        axis.axhline(0, color="#777777", linewidth=1)
        axis.set_title(stage, fontsize=9)
        axis.tick_params(axis="x", rotation=30, labelsize=8)
    axes[0, 0].set_ylabel("Net frames made correct (after − before)")
    figure.suptitle(f"Where the agents changed the VLM's cue, across {n_seeds} seeds", y=1.02)
    figure.tight_layout()
    _save(figure, plots_dir / "change_attribution_across_seeds.png")

    # per-class F1 change
    figure, axes = plt.subplots(1, len(PRIMITIVES), figsize=(3.0 * len(PRIMITIVES), 4.4), squeeze=False)
    for axis, label in zip(axes[0], PRIMITIVES):
        selected = classes.loc[classes["label"] == label]
        strip_box(axis, [selected.loc[selected["run_name"] == m, "delta_f1"].to_numpy() for m in variants], [SHORT[m] for m in variants], colors)
        axis.axhline(0, color="#777777", linewidth=1)
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=30, labelsize=8)
    axes[0, 0].set_ylabel("F1 change vs VLM (all positions)")
    figure.suptitle(f"Per-class F1 change vs VLM, across {n_seeds} seeds", y=1.02)
    figure.tight_layout()
    _save(figure, plots_dir / "class_f1_delta_across_seeds.png")

    # counts: subject-total count MAE per seed (VLM and modes), and per-primitive change
    modes = [BASE, *variants]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    strip_box(axes[0], [by_seed(seed_level, m, "count_mae").to_numpy() for m in modes], [SHORT[m] for m in modes], [plots.mode_color(m) for m in modes])
    axes[0].set(ylabel="Count MAE per subject (lower is better)", title="Total primitive-count error")
    primitives = [p for p in (*PRIMITIVES, "total") if p in set(per_seed["primitive"])]
    width = 0.8 / len(variants)
    for index, mode in enumerate(variants):
        selected = per_seed.loc[per_seed["run_name"] == mode].groupby("primitive")["mean_abs_error_change"].agg(["mean", "std"]).reindex(primitives)
        offsets = np.arange(len(primitives)) + (index - (len(variants) - 1) / 2) * width
        axes[1].bar(offsets, selected["mean"], width=width, yerr=selected["std"], capsize=3, color=plots.mode_color(mode), label=SHORT[mode])
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].set_xticks(np.arange(len(primitives)), primitives)
    axes[1].set(ylabel="Mean change in |count error| vs VLM (negative is better)", title="Count-error change per primitive (mean ± SD over seeds)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    _save(figure, plots_dir / "count_error_across_seeds.png")

    # safety vs accuracy, one point per seed and mode
    figure, axis = plt.subplots(figsize=(7, 5))
    for mode in variants:
        accuracy = by_seed(seed_level, mode, "accuracy") - by_seed(seed_level, BASE, "accuracy")
        unsafe = by_seed(seed_level, BASE, "unsafe_recommendation_rate") - by_seed(seed_level, mode, "unsafe_recommendation_rate")
        axis.scatter(unsafe, accuracy, s=45, color=plots.mode_color(mode), label=SHORT[mode], edgecolor="white", zorder=3)
    axis.axhline(0, color="#777777", linewidth=1)
    axis.axvline(0, color="#777777", linewidth=1)
    axis.set(xlabel="Unsafe-rate reduction vs VLM (right is safer)", ylabel="Accuracy change vs VLM (up is better)",
             title=f"Safety gained vs accuracy paid, one point per seed ({n_seeds} seeds)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, plots_dir / "safety_tradeoff_across_seeds.png")

    # per-subject accuracy change
    subject_ids = sorted(subjects["subject_id"].unique())
    figure, axis = plt.subplots(figsize=(max(7, 0.9 * len(subject_ids) + 2), 4.6))
    for index, mode in enumerate(variants):
        selected = subjects.loc[subjects["run_name"] == mode].groupby("subject_id")["delta_accuracy"].agg(["mean", "std"]).reindex(subject_ids)
        offsets = np.arange(len(subject_ids)) + (index - (len(variants) - 1) / 2) * width
        axis.bar(offsets, selected["mean"], width=width, yerr=selected["std"], capsize=3, color=plots.mode_color(mode), label=SHORT[mode])
    axis.axhline(0, color="#777777", linewidth=1)
    axis.set_xticks(np.arange(len(subject_ids)), subject_ids, rotation=45)
    axis.set(ylabel="Accuracy change vs VLM", title=f"Per-subject accuracy change (mean ± SD over {n_seeds} seeds)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, plots_dir / "subject_accuracy_delta_across_seeds.png")

    ablations = [mode for mode in ABLATIONS if mode in variants] if FULL in variants else []
    if ablations:
        ablation_section(seed_level, ablations, tables_dir, plots_dir, rng)
    exercise_limit_section(runs, [BASE, *variants], tables_dir, plots_dir, rng)


def ablation_section(seed_level: pd.DataFrame, ablations: list[str], tables_dir: Path, plots_dir: Path, rng) -> None:
    """13 and ablation_effects.png: each ablation minus the full MAS."""
    rows = []
    for mode in ablations:
        for column, label, higher in ABLATION_METRICS:
            if column not in seed_level:
                continue
            rows.append(
                {
                    "Comparison": f"{plots.MODE_LABELS[mode]} - {plots.MODE_LABELS[FULL]}",
                    "Metric": label,
                    "metric_column": column,
                    **paired_stats(by_seed(seed_level, FULL, column), by_seed(seed_level, mode, column), rng, higher),
                }
            )
    table = add_holm(pd.DataFrame(rows), ["Comparison"])
    table.to_csv(tables_dir / "13_ablation_vs_mas_across_seeds.csv", index=False, float_format="%.6f")

    metrics = [(c, l, h) for c, l, h in ABLATION_METRICS if c in seed_level]
    columns = 4
    rows_n = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows_n, columns, figsize=(4.4 * columns, 4.2 * rows_n), squeeze=False)
    for axis, (column, label, higher) in zip(axes.flat, metrics):
        groups = [(by_seed(seed_level, mode, column) - by_seed(seed_level, FULL, column)).dropna().to_numpy() for mode in ablations]
        strip_box(axis, groups, [SHORT[m] for m in ablations], [plots.mode_color(m) for m in ablations])
        axis.axhline(0, color="#777777", linewidth=1)
        axis.set_title(f"{label}\n({'higher' if higher else 'lower'} is better)", fontsize=10)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    axes[0, 0].set_ylabel("Change vs full MAS (per seed)")
    figure.suptitle(f"What each agent contributes: ablation minus full MAS, {seed_level['seed'].nunique()} seeds", y=1.02)
    figure.tight_layout()
    _save(figure, plots_dir / "ablation_effects.png")


def exercise_limit_section(runs: dict[int, Path], modes: list[str], tables_dir: Path, plots_dir: Path, rng) -> None:
    """14 and exercise_limit.png: every mode on each patient's first N exercises."""
    frames = []
    for seed, evaluation in runs.items():
        path = evaluation / "exercise_limit.csv"
        if path.is_file():
            frames.append(pd.read_csv(path, dtype={"limit": str}).assign(seed=seed))
    if not frames:
        print("No exercise_limit.csv found; skipping the history-length section")
        return
    limits = pd.concat(frames, ignore_index=True)
    limits = limits[limits["mode"].isin(modes)]
    columns = [column for column, _, _ in LIMIT_METRICS]

    marked = limits[limits["marked"].astype(bool)]
    summary = marked.groupby(["mode", "limit"], sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index()
    rows = []
    contrasts = [(BASE, mode) for mode in modes if mode != BASE]
    if FULL in modes and "mas_no_memory" in modes:
        contrasts.append(("mas_no_memory", FULL))
    for limit in dict.fromkeys(marked["limit"]):
        at = marked[marked["limit"] == limit].rename(columns={"mode": "run_name"})
        for base, variant in contrasts:
            for column, label, higher in LIMIT_METRICS:
                rows.append(
                    {
                        "limit": limit,
                        "Comparison": f"{plots.MODE_LABELS[variant]} - {plots.MODE_LABELS[base]}",
                        "Metric": label,
                        "metric_column": column,
                        **paired_stats(by_seed(at, base, column), by_seed(at, variant, column), rng, higher),
                    }
                )
    paired = add_holm(pd.DataFrame(rows), ["limit", "Comparison"])
    with (tables_dir / "14_exercise_limit_across_seeds.csv").open("w") as handle:
        handle.write("# Part 1: mean and SD over seeds per mode and history length\n")
        summary.to_csv(handle, index=False, float_format="%.6f")
        handle.write("\n# Part 2: paired contrasts over seeds at each marked history length\n")
        paired.to_csv(handle, index=False, float_format="%.6f")

    curve = limits[(limits["limit"] != "all") & limits["complete"].astype(bool)].copy()
    curve["n"] = curve["limit"].astype(int)
    everything = limits[limits["limit"] == "all"]
    marks = sorted({int(v) for v in marked["limit"] if v != "all"})
    right = int(curve["n"].max()) if len(curve) else 1
    all_x = right * 1.6
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for axis, (column, label, higher) in zip(axes.flat, LIMIT_METRICS):
        for mode in modes:
            color = plots.mode_color(mode)
            stats_n = curve[curve["mode"] == mode].groupby("n")[column].agg(["mean", "std"])
            axis.plot(stats_n.index, stats_n["mean"], color=color, linewidth=1.6, label=plots.MODE_LABELS[mode])
            axis.fill_between(stats_n.index, stats_n["mean"] - stats_n["std"].fillna(0), stats_n["mean"] + stats_n["std"].fillna(0),
                              color=color, alpha=0.12, linewidth=0)
            picked = stats_n.reindex(marks)
            axis.plot(picked.index, picked["mean"], "o", color=color, markersize=5)
            final = everything[everything["mode"] == mode][column]
            if len(final):
                axis.errorbar([all_x], [final.mean()], yerr=[final.std() if len(final) > 1 else 0], fmt="D", color=color, markersize=6, capsize=3)
        axis.set_xscale("log")
        axis.minorticks_off()
        ticks = [*marks, all_x]
        axis.set_xticks(ticks, [*map(str, marks), "all"])
        axis.set(xlabel="Exercises per patient (first N; 'all' = every exercise)", title=f"{label} ({'higher' if higher else 'lower'} is better)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"History length: each patient's first N exercises (mean ± SD over {limits['seed'].nunique()} seeds); "
        "memory grows as exercises are added",
        y=1.01,
    )
    figure.tight_layout()
    _save(figure, plots_dir / "exercise_limit.png")


# ------------------------------------------------------------------ main


def main() -> None:
    args = parse_args()
    plots._style()
    rng = np.random.default_rng(RNG_SEED)
    runs = seed_runs(args.root, args.split)
    modes = present_modes(runs)
    variants = [mode for mode in modes if mode != BASE]
    output = args.output or args.root / "report"
    tables_dir, plots_dir = output / "tables", output / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Seeds ({len(runs)}): {list(runs)}; modes: {modes}")

    rows = []
    for seed, evaluation in runs.items():
        comparison = pd.read_csv(evaluation / "comparison.csv").set_index("mode")
        rows += [seed_level_row(seed, mode, evaluation, args.split, comparison.loc[mode]) for mode in modes]
    seed_level = pd.DataFrame(rows)
    seed_level.to_csv(output / "seed_level_metrics.csv", index=False, float_format="%.6f")

    task_long = load_stacked(runs, modes, "final/tables/task_metrics.csv")
    per_class_long = load_stacked(runs, modes, "final/tables/primitive_per_class_metrics.csv")
    uefma_long = load_stacked(runs, modes, f"predictions/{args.split}_ue_fma_subject_predictions.csv")
    task_current = task_long.loc[task_long["prediction_position"] == POSITION]
    tasks = list(task_current.groupby("task_name")["support"].first().sort_values(ascending=False).index)
    classes = sorted(per_class_long.loc[per_class_long["prediction_position"] == POSITION, "label"].unique())
    uefma_r = uefma_by_seed(uefma_long)
    n_seeds, n_subjects = len(runs), int(uefma_long["subject_id"].nunique())

    primitive_stats = primitive_table(seed_level, variants, rng)
    uefma_stats = uefma_table(uefma_r, modes, variants, rng)
    tables = {
        "01_primitive_significance_across_seeds.csv": primitive_stats,
        "02_variant_spread_across_seeds.csv": spread_table(seed_level, modes, rng),
        "03_task_significance_across_seeds.csv": task_table(task_current, tasks, modes, variants, rng),
        "04_per_class_significance_across_seeds.csv": per_class_table(
            per_class_long.loc[per_class_long["prediction_position"] == POSITION], classes, modes, variants, rng
        ),
        "05_uefma_correlation_across_seeds.csv": uefma_stats,
        "06_task_metrics_long.csv": task_long[
            ["seed", "run_name", "variant", "task_name", "prediction_position", "support", "accuracy", "macro_f1",
             "weighted_f1", "reach_f1", "reposition_f1", "event_level_f1"]
        ].sort_values(["task_name", "prediction_position", "variant", "seed"]),
        "07_uefma_pearson_r_by_seed.csv": uefma_r,
    }
    sequence_metrics = [metric for metric in SEQUENCE_METRICS if metric[0] in seed_level]  # older runs lack them
    sequence_stats = primitive_table(seed_level, variants, rng, sequence_metrics) if sequence_metrics and variants else None
    if sequence_stats is not None:
        tables["15_sequence_significance_across_seeds.csv"] = sequence_stats
    for name, frame in tables.items():
        frame.to_csv(tables_dir / name, index=False, float_format="%.6f")
    # Training reported only the test split, so the agreement check exists only there.
    if args.split == "test":
        training_agreement(seed_level, runs, args.training_seed_metrics, args.training_run_name).to_csv(
            output / "training_agreement.csv", index=False, float_format="%.6f"
        )
    else:
        print(f"Split {args.split!r}: training never scored these subjects; skipping training_agreement.csv")

    plot_metric_spread(seed_level, modes, PRIMITIVE_METRICS, plots_dir / "metric_spread_by_variant.png")
    plot_metric_spread(seed_level, modes, (*PRIMITIVE_METRICS, *MAS_METRICS), plots_dir / "metric_spread_by_variant_with_pvalues.png", primitive_stats)
    plot_paired_deltas(seed_level, variants, (*PRIMITIVE_METRICS, *MAS_METRICS), plots_dir / "paired_delta_by_seed.png")
    plot_metric_spread(seed_level, modes, sequence_metrics, plots_dir / "sequence_metrics_across_seeds.png", sequence_stats, ncols=3)
    plot_task_bars(task_current, tasks, modes, n_seeds, plots_dir / "task_metrics_by_variant.png")
    plot_task_spread(task_current, tasks, modes, n_seeds, plots_dir / "task_metric_spread_across_seeds.png")
    plot_uefma_scatter(uefma_long, uefma_r, modes, n_seeds, plots_dir / "uefma_actual_vs_predicted.png")
    plot_uefma_r(uefma_r, uefma_stats, modes, variants, n_subjects, n_seeds, plots_dir / "uefma_pearson_r_across_seeds.png")

    if variants:
        mas_only(runs, variants, seed_level, output / "mas_only", rng)

    headline = primitive_stats[primitive_stats["metric_column"].isin(["accuracy", "macro_f1", "accuracy_future_primitive", "count_mae", "unsafe_recommendation_rate"])]
    print(headline[["Comparison", "Metric", "Base -> variant", "Delta (bootstrap 95% CI)", "Permutation p", "Holm p (permutation)", "Improved"]].to_string(index=False))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
