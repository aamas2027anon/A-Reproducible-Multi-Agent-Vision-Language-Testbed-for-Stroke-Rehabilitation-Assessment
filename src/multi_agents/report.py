"""Multi-seed MAS measurements, diagnostics and paired ablation comparisons."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from multi_agents.settings import LOG_DIR
import config as train_config
from sign_flip import sign_flip_pvalue
from run_log import start_log
from multi_agents.metrics import reference_agreement

VLM, MAS = "vlm", "mas"
MODES = ("vlm", "mas", "mas_no_memory", "mas_no_planning", "mas_no_safety")
ABLATIONS = ("mas_no_memory", "mas_no_planning", "mas_no_safety")
MODE_NAMES = {"vlm": "VLM", "mas": "MAS", "mas_no_memory": "-memory", "mas_no_planning": "-planning", "mas_no_safety": "-safety"}
# Metrics where lower is better; every other metric is better higher.
LOWER_IS_BETTER = frozenset({
    "action_error_rate", "count_mae", "count_mape", "ue_fma_exercise_mae", "ue_fma_mae", "ue_fma_rmse",
    "unsafe_rate", "implausible_transition_rate", "withheld_correct_and_safe_rate",
})
# (column of metrics.csv, label, scale)
SUMMARY_METRICS = (
    ("accuracy", "Accuracy (frame)", 1),
    ("weighted_f1", "Weighted F1 (frame)", 1),
    ("edit_score", "Edit score", 1),
    ("action_error_rate", "AER", 1),
    ("count_mae", "Count MAE", 1),
    ("count_mape", "MAPE (%)", 1),
    ("ue_fma_exercise_mae", "UE-FMA MAE, exercise", 1),
    ("ue_fma_mae", "UE-FMA MAE, subject", 1),
    ("unsafe_rate", "Unsafe rate (%)", 100),
    ("coverage", "Coverage (% frames)", 100),
)
ADDITIONAL_METRICS = (
    ("macro_f1", "Macro F1 (frame)", 1),
    ("accuracy_h0", "Accuracy (window, h=0)", 1),
    ("weighted_f1_h0", "Weighted F1 (window, h=0)", 1),
    ("macro_f1_h0", "Macro F1 (window, h=0)", 1),
    ("accuracy_h4", "Future-primitive acc. (h=4)", 1),
    ("weighted_f1_h4", "Future-primitive weighted F1 (h=4)", 1),
    ("macro_f1_h4", "Future-primitive macro F1 (h=4)", 1),
    ("count_pearson_r", "Count Pearson r", 1),
    ("ue_fma_rmse", "UE-FMA RMSE, subject", 1),
    ("ue_fma_pearson_r", "UE-FMA Pearson r, subject", 1),
    ("implausible_transition_rate", "Implausible transitions (%)", 100),
)
ABLATION_METRICS = (
    ("weighted_f1", "Weighted F1 (frame)"),
    ("accuracy", "Accuracy (frame)"),
    ("accuracy_h4", "Future-primitive acc. (h=4)"),
    ("count_mae", "Count MAE"),
    ("edit_score", "Edit score (frame)"),
    ("unsafe_rate", "Unsafe rate (fraction)"),
    ("coverage", "Coverage (fraction)"),
    ("withheld_correct_and_safe_rate", "Unnecessary escalation rate"),
    ("ue_fma_exercise_mae", "UE-FMA MAE, exercise"),
)
ABLATION_TABLES = {
    "mas_no_memory": "no_memory_vs_mas.csv",
    "mas_no_planning": "no_planning_vs_mas.csv",
    "mas_no_safety": "no_safety_vs_mas.csv",
}
TUNED_SETTINGS = (
    ("emission_temperature", "tau"),
    ("transition_strength", "lambda"),
    ("population_prior_strength", "kappa"),
    ("rest_cost", "rho"),
    ("ue_calib_a", "beta0"),
    ("ue_calib_b", "beta1"),
)
COUNT_ROWS = ("reach", "reposition", "rest", "stabilize", "transport", "total")
STAGE_NAMES = {"planning": "Planning", "safety": "Safety"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=train_config.MAS_RUN_DIR, help="Directory with seed_<S>/ runs")
    parser.add_argument("--output", type=Path, default=train_config.REPORT_DIR, help="Default: outputs/report")
    parser.add_argument("--training-root", type=Path, default=None)
    return parser.parse_args()


def seed_runs(root: Path) -> dict[int, Path]:
    runs = {}
    for directory in root.glob("seed_*"):
        match = re.fullmatch(r"seed_(\d+)", directory.name)
        if (match and int(match.group(1)) in train_config.SEEDS
                and (directory / "eval_test" / "metrics.csv").is_file()):
            runs[int(match.group(1))] = directory
    if not runs:
        raise SystemExit(f"No seed_<S>/eval_test/metrics.csv under {root}")
    return {seed: runs[seed] for seed in train_config.SEEDS if seed in runs}


def stack(runs: dict[int, Path], name: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(directory / "eval_test" / name).assign(seed=seed) for seed, directory in runs.items()],
        ignore_index=True,
    )


def by_seed(frame: pd.DataFrame, mode: str, column: str) -> pd.Series:
    return frame.loc[frame["mode"] == mode].set_index("seed")[column].sort_index()


def better(differences: pd.Series, column: str) -> pd.Series:
    return differences < 0 if column in LOWER_IS_BETTER else differences > 0


def worse(differences: pd.Series, column: str) -> pd.Series:
    return differences > 0 if column in LOWER_IS_BETTER else differences < 0


def mode_columns(seed_level: pd.DataFrame, column: str, scale: float) -> dict[str, float]:
    row = {}
    for mode in MODES:
        values = by_seed(seed_level, mode, column) * scale
        row[f"{MODE_NAMES[mode]} mean"] = values.mean()
        row[f"{MODE_NAMES[mode]} SD"] = values.std(ddof=1)
    return row


def mas_seeds_improved(seed_level: pd.DataFrame, column: str) -> int:
    differences = by_seed(seed_level, MAS, column) - by_seed(seed_level, VLM, column)
    return int(better(differences, column).sum())


def comparison_table(seed_level: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": label, "n seeds": int(seed_level["seed"].nunique()), **mode_columns(seed_level, column, scale), "MAS seeds improved": mas_seeds_improved(seed_level, column)}
        for column, label, scale in SUMMARY_METRICS
    ])


def tuned_settings(runs: dict[int, Path]) -> pd.DataFrame:
    rows = []
    for seed, directory in runs.items():
        config = json.loads((directory / "tuned_config.json").read_text(encoding="utf-8"))
        rows.append({"Seed": seed, **{name: config[key] for key, name in TUNED_SETTINGS}})
    return pd.DataFrame(rows)


def additional_metrics(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label, scale in ADDITIONAL_METRICS:
        row = {"Metric": label, "n seeds": int(seed_level["seed"].nunique()), **mode_columns(seed_level, column, scale)}
        row["Relative delta (%)"] = (
            100 * (row["MAS mean"] - row["VLM mean"]) / row["VLM mean"]
            if row["VLM mean"] != 0 else np.nan
        )
        row["MAS seeds improved"] = mas_seeds_improved(seed_level, column)
        rows.append(row)
    return pd.DataFrame(rows)


def primitive_counts(seed_level: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Count MAE per primitive: the mean over subjects per seed, then mean and SD over seeds."""
    per_seed = counts.groupby(["mode", "seed", "primitive"], as_index=False)["abs_error"].mean()
    rows = []
    for primitive in COUNT_ROWS:
        selected = per_seed[per_seed["primitive"] == primitive]
        row = {"Primitive": primitive, "n seeds": int(seed_level["seed"].nunique())}
        for mode in MODES:
            values = by_seed(selected, mode, "abs_error")
            row[f"{MODE_NAMES[mode]} mean"], row[f"{MODE_NAMES[mode]} SD"] = values.mean(), values.std(ddof=1)
        rows.append(row)
    rows.append({"Primitive": "MAPE, total (%)", "n seeds": int(seed_level["seed"].nunique()), **mode_columns(seed_level, "count_mape", 1)})
    return pd.DataFrame(rows)


def agent_diagnostics(seed_level: pd.DataFrame, attribution: pd.DataFrame) -> pd.DataFrame:
    """Shares and accuracies in percent; NaN marks a stage that is disabled in that mode."""
    modes = [mode for mode in MODES if mode != VLM]
    rows: dict[str, dict[str, float]] = {}
    for mode in modes:
        for stage, name in STAGE_NAMES.items():
            selected = attribution[(attribution["mode"] == mode) & (attribution["stage"] == stage)]
            active = selected["frames_changed"].mean() != 0
            values = {
                f"{name}: frames changed (%) mean": 100 * selected["share_of_frames"].mean(),
                f"{name}: frames changed (%) SD": 100 * selected["share_of_frames"].std(ddof=1),
                f"{name}: accuracy on changed frames before (%)": 100 * selected["accuracy_before"].mean(),
                f"{name}: accuracy on changed frames after (%)": 100 * selected["accuracy_after"].mean(),
            }
            for key, value in values.items():
                rows.setdefault(key, {})[mode] = value if active else np.nan
        withheld = by_seed(seed_level, mode, "withheld_frames")
        correct = by_seed(seed_level, mode, "withheld_correct_and_safe_rate") * 100
        any_withheld = withheld.sum() > 0
        rows.setdefault("Withheld frames per seed mean", {})[mode] = withheld.mean()
        rows.setdefault("Withheld frames per seed SD", {})[mode] = withheld.std(ddof=1)
        rows.setdefault("Withheld frames that were correct and safe (%) mean", {})[mode] = correct.mean() if any_withheld else np.nan
        rows.setdefault("Withheld frames that were correct and safe (%) SD", {})[mode] = correct.std(ddof=1) if any_withheld else np.nan
        rows.setdefault("Released primitives violating C1", {})[mode] = int(by_seed(seed_level, mode, "released_violating_c1").sum())
    table = pd.DataFrame(rows).T[modes]
    table.columns = [MODE_NAMES[mode] for mode in modes]
    table = table.rename_axis("Diagnostic").reset_index()
    table.insert(1, "n seeds", int(seed_level["seed"].nunique()))
    return table


def ablation_table(seed_level: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for column, label in ABLATION_METRICS:
        base, variant = by_seed(seed_level, MAS, column), by_seed(seed_level, mode, column)
        differences = variant - base.loc[variant.index]
        rows.append({
            "Metric": label, "n seeds": len(differences),
            "MAS mean": base.mean(), "MAS SD": base.std(ddof=1),
            f"{MODE_NAMES[mode]} mean": variant.mean(), f"{MODE_NAMES[mode]} SD": variant.std(ddof=1),
            "Delta": differences.mean(),
            "p": sign_flip_pvalue(differences.to_numpy()),
            "Better": int(better(differences, column).sum()),
            "Worse": int(worse(differences, column).sum()),
        })
    return pd.DataFrame(rows)


def aggregation_table(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, column, label in (
        (VLM, "ue_fma_exercise_mae", "Isolated exercise mean"),
        (VLM, "ue_fma_running_mean_mae", "Unweighted running mean"),
        (MAS, "ue_fma_exercise_mae", "Precision-weighted running estimate"),
    ):
        values = by_seed(seed_level, mode, column) if column in seed_level else pd.Series(dtype=float)
        rows.append({"Estimate": label, "n seeds": int(values.count()),
                     "MAE mean": values.mean(), "MAE SD": values.std(ddof=1),
                     "Status": "available" if len(values) and values.notna().all() else "unavailable"})
    return pd.DataFrame(rows)


def agreement_table(runs: dict[int, Path], training_root: Path) -> pd.DataFrame:
    rows = []
    text_run = next(name for name, mode in train_config.VARIANTS if mode == "text")
    for seed, directory in runs.items():
        reference = training_root / "seeds" / f"seed_{seed}" / text_run / "predictions" / "test_combined_window_predictions.csv"
        replay = directory / "eval_test" / "primitive_predictions.csv"
        row = {"seed": seed, "status": "unavailable", "replay_frames": np.nan,
               "reference_frames": np.nan, "matched_frames": np.nan, "agreement": np.nan}
        if replay.is_file() and reference.is_file():
            predictions = pd.read_csv(replay)
            predictions = predictions[predictions["mode"] == VLM]
            row.update(reference_agreement(predictions, pd.read_csv(reference)))
        rows.append(row)
    return pd.DataFrame(rows)


def export_report(root: Path, output: Path = train_config.REPORT_DIR,
                  training_root: Path | None = None) -> dict[str, Path]:
    runs = seed_runs(root)
    tables_dir = Path(output) / "tables"
    seed_level = stack(runs, "metrics.csv")
    if seed_level.duplicated(["seed", "mode"]).any():
        raise ValueError("Duplicate seed/mode measurements")
    for seed, values in seed_level.groupby("seed"):
        missing = set(MODES) - set(values["mode"])
        if missing:
            raise ValueError(f"Seed {seed} is missing modes: {sorted(missing)}")
    seed_level = seed_level[["seed", *[c for c in seed_level if c != "seed"]]].sort_values(["seed", "mode"])
    print(f"MAS seeds ({len(runs)}): {list(runs)}")
    training_root = training_root or train_config.TRAINING_RUN_DIR
    if len(runs) == 1:
        seed = next(iter(runs))
        single_dir = Path(output) / f"single_seed_{seed}"
        single_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "mas_metrics": single_dir / "mas_metrics.csv",
            "prediction_agreement": single_dir / "prediction_agreement.csv",
        }
        seed_level.to_csv(outputs["mas_metrics"], index=False, float_format="%.6f")
        agreement_table(runs, training_root).to_csv(
            outputs["prediction_agreement"], index=False, float_format="%.6f"
        )
        return outputs
    tables = {
        "mas_seed_metrics.csv": seed_level,
        "mas_vs_vlm.csv": comparison_table(seed_level),
        "tuned_settings.csv": tuned_settings(runs),
        "mas_additional_metrics.csv": additional_metrics(seed_level),
        "primitive_counts.csv": primitive_counts(seed_level, stack(runs, "count_errors.csv")),
        "agent_diagnostics.csv": agent_diagnostics(seed_level, stack(runs, "change_attribution.csv")),
        "uefma_aggregation.csv": aggregation_table(seed_level),
        "prediction_agreement.csv": agreement_table(runs, training_root),
        **{filename: ablation_table(seed_level, mode) for mode, filename in ABLATION_TABLES.items()},
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, table in tables.items():
        outputs[Path(name).stem] = tables_dir / name
        table.to_csv(tables_dir / name, index=False, float_format="%.6f")
    print(tables["mas_vs_vlm.csv"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return outputs


def main() -> None:
    args = parse_args()
    start_log(LOG_DIR, "report")
    export_report(args.root, args.output, args.training_root)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
