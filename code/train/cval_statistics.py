from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, rankdata, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from artifacts import ALIGN_COLUMNS
from config import (
    BASE_RUN_NAME,
    CLASS_NAMES,
    CONDITIONING_NONE,
    EXPECTED_TRAIN_SUBJECTS,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    RuntimeConfig,
    VARIANTS,
)


DISPLAY_NAMES = {
    "qwen_without_patient_conditioning": "Base",
    "qwen_with_patient_text": "Base + patient text",
    "qwen_with_patient_prefix": "Base + patient prefix",
}
BASE_NAME = DISPLAY_NAMES[BASE_RUN_NAME]
# Each conditioning variant is contrasted against the unconditioned base; the two
# conditionings are not compared against each other.
CONDITIONED_RUN_NAMES = tuple(
    run_name for run_name, mode in VARIANTS if mode != CONDITIONING_NONE
)
METRICS = (
    ("weighted_f1", "Weighted F1"),
    ("accuracy", "Accuracy"),
    ("reach_f1", "Reach F1"),
    ("reposition_f1", "Reposition F1"),
)


def _load_oof_assignments(config: RuntimeConfig) -> pd.DataFrame:
    path = config.output_root / "fold_assignments.csv"
    assignments = pd.read_csv(path)[["subject_id", "fold_id"]]
    assignments = assignments.sort_values("subject_id").reset_index(drop=True)
    if assignments.empty or not assignments["subject_id"].is_unique:
        raise AssertionError("OOF assignments must contain one row per held-out subject")
    if assignments["fold_id"].nunique() != config.n_folds:
        raise AssertionError("OOF assignments do not match configured fold count")
    return assignments


def per_subject_primitive_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, subject_frame in frame.groupby("subject_id", sort=True):
        actual = subject_frame["actual_primitive"].astype(str)
        predicted = subject_frame["predicted_primitive"].astype(str)
        label_f1 = precision_recall_fscore_support(
            actual,
            predicted,
            labels=["reach", "reposition"],
            zero_division=0,
        )[2]
        rows.append(
            {
                "subject_id": subject_id,
                "weighted_f1": f1_score(
                    actual,
                    predicted,
                    labels=list(CLASS_NAMES),
                    average="weighted",
                    zero_division=0,
                ),
                "accuracy": accuracy_score(actual, predicted),
                "reach_f1": label_f1[0],
                "reposition_f1": label_f1[1],
            }
        )
    result = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    if len(result) != EXPECTED_TRAIN_SUBJECTS or not result["subject_id"].is_unique:
        raise AssertionError("Expected exactly 32 unique per-subject metric rows")
    return result


def monte_carlo_sign_flip_pvalue(
    differences: pd.Series | np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[float, int]:
    diffs = np.asarray(differences, dtype=float)
    if np.allclose(diffs, 0):
        return 1.0, n_permutations
    observed = abs(float(diffs.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_permutations, len(diffs)))
    permuted = np.abs((signs * diffs).mean(axis=1))
    extreme = int(np.count_nonzero(permuted >= observed - 1e-12))
    return float((extreme + 1) / (n_permutations + 1)), extreme


def bootstrap_mean_ci(
    differences: pd.Series | np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    means = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def paired_cohens_d(differences: pd.Series | np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    sd = values.std(ddof=1)
    return np.nan if np.isclose(sd, 0) else float(values.mean() / sd)


def wilcoxon_two_sided(differences: pd.Series | np.ndarray) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if np.allclose(values, 0):
        return 0.0, 1.0
    result = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def matched_pairs_rank_biserial(differences: pd.Series | np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    nonzero = values[~np.isclose(values, 0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def _mcnemar(
    base: pd.DataFrame,
    prefix: pd.DataFrame,
    metric: str,
) -> dict[str, Any]:
    if metric == "weighted_f1":
        return {
            "base_wrong_prefix_right": pd.NA,
            "base_right_prefix_wrong": pd.NA,
            "discordant": pd.NA,
            "pvalue": np.nan,
            "result": "N/A: McNemar tests correctness, not weighted F1",
        }
    if metric == "accuracy":
        base_correct = base["actual_primitive"].eq(base["predicted_primitive"])
        prefix_correct = prefix["actual_primitive"].eq(prefix["predicted_primitive"])
        suffix = "overall correctness"
    else:
        label = "reach" if metric == "reach_f1" else "reposition"
        actual_binary = base["actual_primitive"].eq(label)
        base_correct = base["predicted_primitive"].eq(label).eq(actual_binary)
        prefix_correct = prefix["predicted_primitive"].eq(label).eq(actual_binary)
        suffix = f"{label}-vs-rest correctness"
    base_wrong_prefix_right = int((~base_correct & prefix_correct).sum())
    base_right_prefix_wrong = int((base_correct & ~prefix_correct).sum())
    discordant = base_wrong_prefix_right + base_right_prefix_wrong
    pvalue = (
        float(
            binomtest(
                min(base_wrong_prefix_right, base_right_prefix_wrong),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "base_wrong_prefix_right": base_wrong_prefix_right,
        "base_right_prefix_wrong": base_right_prefix_wrong,
        "discordant": discordant,
        "pvalue": pvalue,
        "result": f"{pvalue:.6e} ({suffix}; discordant n={discordant})",
    }


def _observed_conclusion(mean_difference: float, permutation_p: float) -> str:
    if permutation_p < 0.05 and mean_difference > 0:
        return "Significant improvement with patient prefix (Monte Carlo paired p < 0.05)."
    if permutation_p < 0.05 and mean_difference < 0:
        return "Significant decrease with patient prefix (Monte Carlo paired p < 0.05)."
    return "No statistically significant paired difference (Monte Carlo paired p >= 0.05)."


def _load_aligned_primitive_frames(config: RuntimeConfig) -> dict[str, pd.DataFrame]:
    frames = {}
    reference_truth = None
    expected_subjects = set(_load_oof_assignments(config)["subject_id"])
    for run_name, _ in VARIANTS:
        path = (
            config.output_root
            / "oof"
            / run_name
            / "predictions"
            / "oof_combined_window_predictions.csv"
        )
        frame = pd.read_csv(path).sort_values(ALIGN_COLUMNS).reset_index(drop=True)
        if set(frame["subject_id"]) != expected_subjects:
            raise AssertionError(f"{run_name} OOF subjects do not match assignments")
        if set(frame["horizon"]) != set(range(5)):
            raise AssertionError(f"{run_name} does not contain all five horizons")
        if frame.duplicated(ALIGN_COLUMNS).any():
            raise AssertionError(f"Duplicate primitive keys in {run_name}")
        truth = frame[[*ALIGN_COLUMNS, "actual_primitive"]]
        if reference_truth is None:
            reference_truth = truth
        elif not truth.equals(reference_truth):
            raise AssertionError("Base and patient-prefix ground truth are not aligned")
        frames[DISPLAY_NAMES[run_name]] = frame
    return frames


def _primitive_statistics(
    frames: dict[str, pd.DataFrame], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = per_subject_primitive_metrics(frames[BASE_NAME]).add_suffix("_base")
    subject = subject.rename(columns={"subject_id_base": "subject_id"})
    for run_name in CONDITIONED_RUN_NAMES:
        variant_subject = per_subject_primitive_metrics(frames[DISPLAY_NAMES[run_name]])
        subject = subject.merge(
            variant_subject.add_suffix(f"_{run_name}").rename(
                columns={f"subject_id_{run_name}": "subject_id"}
            ),
            on="subject_id",
            validate="one_to_one",
        )
    rows = []
    for run_name in CONDITIONED_RUN_NAMES:
        variant_name = DISPLAY_NAMES[run_name]
        comparison = f"{variant_name} − {BASE_NAME}"
        for metric, label in METRICS:
            differences = subject[f"{metric}_{run_name}"] - subject[f"{metric}_base"]
            ci_low, ci_high = bootstrap_mean_ci(differences, rng)
            permutation_p, extreme = monte_carlo_sign_flip_pvalue(differences, rng)
            wilcoxon_statistic, wilcoxon_p = wilcoxon_two_sided(differences)
            mcnemar = _mcnemar(frames[BASE_NAME], frames[variant_name], metric)
            base_mean = float(subject[f"{metric}_base"].mean())
            variant_mean = float(subject[f"{metric}_{run_name}"].mean())
            mean_difference = float(differences.mean())
            rows.append(
                {
                    "Comparison": comparison,
                    "Metric": label,
                    "n": len(differences),
                    "Base mean": base_mean,
                    "Variant mean": variant_mean,
                    "Base → variant": f"{base_mean:.3f} → {variant_mean:.3f}",
                    "Mean paired delta": mean_difference,
                    "Bootstrap 95% CI low": ci_low,
                    "Bootstrap 95% CI high": ci_high,
                    "Δ (bootstrap 95% CI)": (
                        f"{mean_difference:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}]"
                    ),
                    "Paired Cohen d": paired_cohens_d(differences),
                    "Permutation method": (
                        "Two-sided paired Monte Carlo sign flips with plus-one correction"
                    ),
                    "Monte Carlo sign flips": N_PERMUTATIONS,
                    "Monte Carlo extreme count": extreme,
                    "Monte Carlo permutation p": permutation_p,
                    "Paired permutation p (10,000 Monte Carlo sign flips; +1 corrected)": (
                        f"({extreme}+1)/({N_PERMUTATIONS}+1) = {permutation_p:.6f}"
                    ),
                    "Subjects improved": int((differences > 0).sum()),
                    "Subjects tied": int(np.isclose(differences, 0).sum()),
                    "Subjects worsened": int((differences < 0).sum()),
                    "Improved": (
                        f"{int((differences > 0).sum())}/{len(differences)} improved"
                    ),
                    "Wilcoxon statistic": wilcoxon_statistic,
                    "Wilcoxon p": wilcoxon_p,
                    "Matched-pairs rank-biserial r": matched_pairs_rank_biserial(
                        differences
                    ),
                    "McNemar base wrong, variant right": mcnemar[
                        "base_wrong_prefix_right"
                    ],
                    "McNemar base right, variant wrong": mcnemar[
                        "base_right_prefix_wrong"
                    ],
                    "McNemar discordant frame n": mcnemar["discordant"],
                    "McNemar exact p": mcnemar["pvalue"],
                    "Frame McNemar result": mcnemar["result"],
                    "Conclusion": _observed_conclusion(mean_difference, permutation_p),
                }
            )
    return pd.DataFrame(rows), subject


def bootstrap_pearson_ci(
    actual: pd.Series,
    predicted: pd.Series,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    indices = np.arange(len(actual_values))
    correlations = []
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        sample_actual = actual_values[sample]
        sample_predicted = predicted_values[sample]
        if np.unique(sample_actual).size < 2 or np.unique(sample_predicted).size < 2:
            continue
        correlations.append(np.corrcoef(sample_actual, sample_predicted)[0, 1])
    if not correlations:
        return np.nan, np.nan
    return tuple(np.quantile(correlations, [0.025, 0.975]).astype(float))


def _uefma_statistics(config: RuntimeConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    reference = None
    expected_subjects = set(_load_oof_assignments(config)["subject_id"])
    for run_name, _ in VARIANTS:
        path = (
            config.output_root
            / "oof"
            / run_name
            / "predictions"
            / "oof_ue_fma_subject_predictions.csv"
        )
        frame = pd.read_csv(path).sort_values("subject_id").reset_index(drop=True)
        if not frame["subject_id"].is_unique or set(frame["subject_id"]) != expected_subjects:
            raise AssertionError("UE-FMA subjects must match persisted OOF assignments")
        truth = frame[["subject_id", "actual_ue_fma"]]
        if reference is None:
            reference = truth
        elif not truth.equals(reference):
            raise AssertionError("Base and prefix UE-FMA truth are not aligned")
        result = pearsonr(frame["actual_ue_fma"], frame["predicted_ue_fma"])
        ci_low, ci_high = bootstrap_pearson_ci(
            frame["actual_ue_fma"], frame["predicted_ue_fma"], rng
        )
        rows.append(
            {
                "Variant": DISPLAY_NAMES[run_name],
                "n": len(frame),
                "Pearson r": result.statistic,
                "Pearson p": result.pvalue,
                "Bootstrap 95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            }
        )
    return pd.DataFrame(rows)


def _counting_statistics(config: RuntimeConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    reference_truth: pd.DataFrame | None = None
    assignments = _load_oof_assignments(config)
    expected_subjects = set(assignments["subject_id"])
    expected_fold_ids = set(assignments["fold_id"])
    for run_name, _ in VARIANTS:
        path = (
            config.output_root
            / "oof"
            / run_name
            / "counting"
            / "predictions"
            / "oof_subject_total_counts.csv"
        )
        frame = pd.read_csv(path).sort_values("subject_id").reset_index(drop=True)
        if not frame["subject_id"].is_unique or set(frame["subject_id"]) != expected_subjects:
            raise AssertionError("Counting subjects must match persisted OOF assignments")
        if set(frame["fold_id"]) != expected_fold_ids:
            raise AssertionError("Counting statistics do not cover persisted OOF folds")
        truth = frame[["subject_id", "actual_count"]]
        if reference_truth is None:
            reference_truth = truth
        else:
            pd.testing.assert_frame_equal(
                truth,
                reference_truth,
                check_dtype=False,
                obj="Base and patient-prefix actual primitive counts",
            )
        actual = frame["actual_count"].astype(float)
        predicted = frame["predicted_count"].astype(float)
        if actual.nunique() > 1 and predicted.nunique() > 1:
            correlation = pearsonr(actual, predicted)
            pearson_r = float(correlation.statistic)
            pearson_p = float(correlation.pvalue)
            ci_low, ci_high = bootstrap_pearson_ci(actual, predicted, rng)
        else:
            pearson_r = pearson_p = ci_low = ci_high = np.nan
        valid_percent = frame["actual_count"].ne(0)
        rows.append(
            {
                "Variant": DISPLAY_NAMES[run_name],
                "n": len(frame),
                "Actual mean count": actual.mean(),
                "Actual count SD": actual.std(ddof=1),
                "Predicted mean count": predicted.mean(),
                "Predicted count SD": predicted.std(ddof=1),
                "Mean predicted percent of actual": frame.loc[
                    valid_percent, "predicted_percent_of_actual"
                ].mean(),
                "Mean signed percent error": frame.loc[
                    valid_percent, "signed_percent_error"
                ].mean(),
                "Mean absolute percent error": frame.loc[
                    valid_percent, "absolute_percent_error"
                ].mean(),
                "Count MAE": mean_absolute_error(actual, predicted),
                "Count RMSE": mean_squared_error(actual, predicted) ** 0.5,
                "Pearson r": pearson_r,
                "Pearson p": pearson_p,
                "Bootstrap 95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            }
        )
    return pd.DataFrame(rows)


def _save_delta_plot(subject: pd.DataFrame, output_root: Path) -> Path:
    """One column of per-subject deltas for each conditioning variant against base."""
    plot_dir = output_root / "oof" / "analysis"
    plot_dir.mkdir(parents=True, exist_ok=True)
    columns = len(CONDITIONED_RUN_NAMES)
    figure, axes = plt.subplots(
        len(METRICS),
        columns,
        figsize=(5.5 * columns, 5.5 * len(METRICS)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (metric, label) in enumerate(METRICS):
        for column_index, run_name in enumerate(CONDITIONED_RUN_NAMES):
            axis = axes[row_index][column_index]
            plot_frame = subject[["subject_id"]].copy()
            plot_frame["delta"] = (
                subject[f"{metric}_{run_name}"] - subject[f"{metric}_base"]
            )
            plot_frame = plot_frame.sort_values("delta")
            colors = np.where(plot_frame["delta"] >= 0, "#2e7d32", "#9b2c2c")
            axis.barh(plot_frame["subject_id"], plot_frame["delta"], color=colors)
            axis.axvline(0, color="#777777", linewidth=1)
            axis.set_title(f"{label} — {DISPLAY_NAMES[run_name]}")
            axis.set_xlabel(f"{DISPLAY_NAMES[run_name]} minus {BASE_NAME}")
            axis.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    figure.suptitle("Per-subject OOF conditioning deltas against base", fontsize=13)
    path = plot_dir / "stroke_rehab_per_subject_base_vs_conditioning_deltas.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def export_crossval_statistics(config: RuntimeConfig) -> dict[str, Path]:
    rng = np.random.default_rng(config.seed)
    frames = _load_aligned_primitive_frames(config)
    primitive_table, subject = _primitive_statistics(frames, rng)
    uefma_table = _uefma_statistics(config, rng)
    counting_table = _counting_statistics(config, rng)
    config.report_root.mkdir(parents=True, exist_ok=True)
    for existing in config.report_root.glob("*.csv"):
        existing.unlink()
    outputs = {
        "primitive": config.report_root / "01_stroke_rehab_primitive_statistics.csv",
        "ue_fma": config.report_root / "02_stroke_rehab_uefma_correlation_ci.csv",
        "counting": config.report_root / "03_stroke_rehab_counting_correlation_ci.csv",
    }
    primitive_table.to_csv(outputs["primitive"], index=False, float_format="%.6f")
    uefma_table.to_csv(outputs["ue_fma"], index=False, float_format="%.6f")
    counting_table.to_csv(outputs["counting"], index=False, float_format="%.6f")
    if sorted(path.name for path in config.report_root.glob("*.csv")) != sorted(
        path.name for path in outputs.values()
    ):
        raise AssertionError("Cross-validation report directory must contain exactly tables 01/02/03")
    _save_delta_plot(subject, config.output_root)
    return outputs
