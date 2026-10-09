#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Hashable
from pathlib import Path
from typing import Any

# This must be set before importing torch. It keeps eager operations on the
# standard ATen path instead of Triton JIT, which requires Python.h at runtime.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
import pandas as pd

from artifacts import aggregate_oof_predictions, generate_pooled_diagnostics
from claims import export_claims_artifacts
from config import (
    BASE_RUN_NAME,
    CV_FOLD_OPTIONS,
    DEFAULT_STAGE,
    FULL_VALIDATION_SUBJECTS,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_WORKERS,
    PER_DEVICE_BATCH_SIZE,
    RuntimeConfig,
    SEED,
    SEEDS,
    STAGE_NAMES,
    TRAINING_VARIANTS,
    VARIANTS,
    default_cache_root,
    default_data_root,
    default_output_root,
    default_raw_dir,
    portable_path,
    stage_spec,
)
from counting import generate_counting_artifacts, stringify_path_tree
from cval_statistics import export_crossval_statistics
from data import (
    create_fold_assignments,
    fold_subjects,
    npz_subjects,
    validate_data_files,
)
from multiseed import (
    load_or_create_resolved_seeds,
    run_multiseed,
    run_seed,
    seed_from_index,
)
from multiseed_statistics import export_multiseed_statistics
from raw_data import build_training_npzs
from ceiling_report import export_ceiling_report
from training import build_tokenizer, run_experiment
from summary import export_all_runs_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen StrokeRehab subject-level cross-validation pipeline"
    )
    parser.add_argument(
        "command",
        choices=(
            "fold",
            "aggregate",
            "full",
            "claims",
            "summary",
            "all",
            "seed",
            "multiseed",
            "multiseed-aggregate",
            "ceiling-report",
        ),
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        default=DEFAULT_STAGE,
        help=(
            "Which run this is. Sets the output root, report directory, epochs and seed "
            "pool from config.STAGES; each is still individually overridable."
        ),
    )
    parser.add_argument("--fold-index", type=int, help="Held-out fold for the fold command")
    parser.add_argument(
        "--seed-index",
        type=int,
        help="Position in the resolved seed list for the seed command",
    )
    parser.add_argument(
        "--seed-value", type=int, help="Explicit seed for the seed command"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help=(
            "Comma-separated candidate seeds for the seed-driven commands. The stage "
            "decides how many seeds the sweep has; duplicates and short lists are "
            "repaired at startup rather than rejected."
        ),
    )
    # These five default to the selected stage rather than to a constant, so they stay
    # None here and are resolved in runtime_config once --stage is known.
    parser.add_argument("--folds", type=int, choices=CV_FOLD_OPTIONS, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Overrides the stage's epoch count. Raising it on a finished run resumes "
            "from last.ckpt and trains the extra epochs."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Merged train/test NPZs, built here from --raw-dir if missing (default: training_data_dir in project_config.toml)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir(),
        help="Raw StrokeRehab tree (video_extracted_features/, kinematics/); default: data_dir in project_config.toml",
    )
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    spec = stage_spec(args.stage)
    if args.folds is None:
        args.folds = spec.n_folds
    if args.epochs is None:
        args.epochs = spec.epochs
    if args.output_root is None:
        args.output_root = default_output_root() / spec.experiment_group
    if args.report_root is None:
        args.report_root = default_output_root() / spec.report_dirname
    return args


def requested_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.seeds:
        return tuple(stage_spec(args.stage).seeds) or tuple(SEEDS)
    return tuple(int(value) for value in args.seeds.split(",") if value.strip())


def runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig(
        data_root=args.data_root.expanduser().absolute(),
        output_root=args.output_root.expanduser().absolute(),
        report_root=args.report_root.expanduser().absolute(),
        cache_root=args.cache_root.expanduser().absolute(),
        model_name=args.model_name,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        n_folds=args.folds,
        stage=args.stage,
    )
    if config.batch_size <= 0 or config.num_workers < 0 or config.epochs <= 0:
        raise ValueError("batch size/epochs must be positive and workers cannot be negative")
    if config.n_folds not in CV_FOLD_OPTIONS:
        raise ValueError(f"folds must be one of {CV_FOLD_OPTIONS}")
    return config


def run_fold(config: RuntimeConfig, fold_id: int, tokenizer: Any | None = None) -> None:
    assignments = create_fold_assignments(config)
    train_subjects, holdout_subjects = fold_subjects(assignments, fold_id)
    if tokenizer is None:
        tokenizer = build_tokenizer(config.model_name)
    fold_root = config.output_root / "folds" / f"fold_{fold_id:02d}"
    fold_results = {}
    for run_name, conditioning in TRAINING_VARIANTS:
        print(
            f"\nFold {fold_id}: {run_name} ({conditioning}) | train={len(train_subjects)} "
            f"holdout={len(holdout_subjects)}"
        )
        fold_results[run_name] = run_experiment(
            config=config,
            tokenizer=tokenizer,
            run_dir=fold_root / run_name,
            run_name=run_name,
            conditioning=conditioning,
            train_npz=config.train_data_path,
            train_subjects=train_subjects,
            validation_subjects=holdout_subjects,
            prediction_npz=config.train_data_path,
            prediction_subjects=holdout_subjects,
            fold_id=fold_id,
            split="oof",
        )
    (fold_root / "fold_manifest.json").write_text(
        json.dumps(
            {
                "fold_complete": True,
                "fold_id": fold_id,
                "fold_seed": config.seed + fold_id,
                "train_subjects": list(train_subjects),
                "holdout_subjects": list(holdout_subjects),
                "variants": fold_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_aggregate(config: RuntimeConfig) -> dict[str, Any]:
    predictions = aggregate_oof_predictions(config)
    counting = generate_pooled_diagnostics(config)
    tables = export_crossval_statistics(config)
    return {
        "predictions": {
            variant: {name: portable_path(path) for name, path in paths.items()}
            for variant, paths in predictions.items()
        },
        "counting": stringify_path_tree(counting),
        "tables": {name: portable_path(path) for name, path in tables.items()},
    }


def run_full(config: RuntimeConfig, tokenizer: Any | None = None) -> dict[str, Any]:
    if tokenizer is None:
        tokenizer = build_tokenizer(config.model_name)
    training_pool = frozenset(npz_subjects(config.train_data_path, config.cache_root))
    missing_validation = FULL_VALIDATION_SUBJECTS - training_pool
    if missing_validation:
        raise ValueError(f"Configured full validation subjects are missing: {sorted(missing_validation)}")
    validation_subjects = tuple(sorted(FULL_VALIDATION_SUBJECTS))
    train_subjects = tuple(sorted(training_pool - FULL_VALIDATION_SUBJECTS))
    test_subjects = npz_subjects(config.test_data_path, config.cache_root)
    if set(validation_subjects) & set(test_subjects):
        raise AssertionError("Full validation and test subjects overlap")
    full_root = config.output_root / "full_train_test"
    results = {}
    counting_results = {}
    for run_name, conditioning in TRAINING_VARIANTS:
        print(
            f"\nFull train/test: {run_name} ({conditioning}) | train={len(train_subjects)} "
            f"validation={len(validation_subjects)} test={len(test_subjects)}"
        )
        results[run_name] = run_experiment(
            config=config,
            tokenizer=tokenizer,
            run_dir=full_root / run_name,
            run_name=run_name,
            conditioning=conditioning,
            train_npz=config.train_data_path,
            train_subjects=train_subjects,
            validation_subjects=validation_subjects,
            prediction_npz=config.test_data_path,
            prediction_subjects=test_subjects,
            fold_id=-1,
            split="test",
        )
        combined_path = Path(results[run_name]["predictions"]["combined"])
        combined = pd.read_csv(combined_path)
        counting_results[run_name] = generate_counting_artifacts(
            combined, full_root / run_name, run_name, "test"
        )

    truth_columns: list[Hashable] = [
        "subject_id", "task_name", "clip_id", "label_id", "primitive", "actual_count"
    ]
    base_counts = pd.read_csv(
        counting_results[BASE_RUN_NAME]["predictions"]["video_primitive"]
    )
    base_truth = (
        base_counts[truth_columns].sort_values(truth_columns[:-1]).reset_index(drop=True)
    )
    for run_name, _ in VARIANTS:
        if run_name == BASE_RUN_NAME:
            continue
        variant_counts = pd.read_csv(
            counting_results[run_name]["predictions"]["video_primitive"]
        )
        pd.testing.assert_frame_equal(
            variant_counts[truth_columns]
            .sort_values(truth_columns[:-1])
            .reset_index(drop=True),
            base_truth,
            check_dtype=False,
            obj=f"{run_name} and {BASE_RUN_NAME} test actual primitive counts",
        )
    (full_root / "full_run_manifest.json").write_text(
        json.dumps(
            {
                "full_run_complete": True,
                "train_subjects": list(train_subjects),
                "validation_subjects": list(validation_subjects),
                "test_subjects": list(test_subjects),
                "variants": results,
                "counting": stringify_path_tree(counting_results),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


SEED_DRIVEN_COMMANDS = frozenset(
    {"seed", "multiseed", "multiseed-aggregate", "ceiling-report"}
)
FOLD_DRIVEN_COMMANDS = frozenset(
    {"fold", "aggregate", "full", "claims", "summary", "all"}
)


def assert_command_matches_stage(command: str, stage: str) -> None:
    """Stop a command from writing into a stage that does not run it.

    Without this, `seed --stage crossval` would create a seeds/ tree inside the
    cross-validation output root, and `fold --stage ceiling` would train folds the
    ceiling test never reports on.
    """
    spec = stage_spec(stage)
    if command in SEED_DRIVEN_COMMANDS and not spec.is_seed_driven:
        raise ValueError(
            f"{command!r} is a seed-driven command but stage {stage!r} has no seeds; "
            "use --stage ceiling or --stage multiseed"
        )
    if command in FOLD_DRIVEN_COMMANDS and spec.is_seed_driven:
        raise ValueError(
            f"{command!r} belongs to the cross-validation stage, not {stage!r}; "
            "use --stage crossval"
        )
    if command == "ceiling-report" and len(spec.seeds) != 1:
        raise ValueError(
            f"'ceiling-report' expects a single-seed stage; {stage!r} has "
            f"{len(spec.seeds)} seeds"
        )


def run_one_seed(config: RuntimeConfig, args: argparse.Namespace) -> dict[str, Any]:
    if args.seed_value is not None and args.seed_index is not None:
        raise ValueError("Pass either --seed-value or --seed-index, not both")
    if args.seed_value is not None:
        seed = int(args.seed_value)
    elif args.seed_index is not None:
        seed = seed_from_index(config, args.seed_index)
    else:
        # A single-seed stage such as the ceiling test needs no index.
        pool = load_or_create_resolved_seeds(config, requested_seeds(args))
        if len(pool) != 1:
            raise ValueError(
                f"stage {config.stage!r} has {len(pool)} seeds; "
                "pass --seed-index or --seed-value"
            )
        seed = pool[0]
    return run_seed(config, seed)


def run_ceiling_report(config: RuntimeConfig, args: argparse.Namespace) -> dict[str, str]:
    seeds = load_or_create_resolved_seeds(config, requested_seeds(args))
    return {name: portable_path(path) for name, path in export_ceiling_report(config, seeds).items()}


def run_multiseed_aggregate(config: RuntimeConfig, args: argparse.Namespace) -> dict[str, str]:
    seeds = load_or_create_resolved_seeds(config, requested_seeds(args))
    outputs = export_multiseed_statistics(config, seeds)
    return {name: portable_path(path) for name, path in outputs.items()}


def run_claims(config: RuntimeConfig) -> dict[str, str]:
    return {
        name: portable_path(path) for name, path in export_claims_artifacts(config).items()
    }


def run_summary(config: RuntimeConfig) -> dict[str, str]:
    all_runs = export_all_runs_summary(config)
    aggregation_manifest = config.output_root / "oof" / "aggregation_manifest.json"
    full_run_manifest = config.output_root / "full_train_test" / "full_run_manifest.json"
    for path in (aggregation_manifest, full_run_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    statistical_tables = {
        "primitive": config.report_root / "01_stroke_rehab_primitive_statistics.csv",
        "ue_fma": config.report_root / "02_stroke_rehab_uefma_correlation_ci.csv",
        "counting": config.report_root / "03_stroke_rehab_counting_correlation_ci.csv",
    }
    for path in statistical_tables.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    claims_root = config.output_root / "claims_analysis_artifacts_crossval"
    claims_outputs = {
        "table1_csv": claims_root / "table1_metrics.csv",
        "table1_tex": claims_root / "table1_metrics.tex",
        "table2_csv": claims_root / "table2_uefma.csv",
        "table2_tex": claims_root / "table2_uefma.tex",
        "figure": claims_root / "uefma.png",
    }
    for path in claims_outputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    counting_roots = {
        run_name: {
            "oof": config.output_root / "oof" / run_name / "counting",
            "test": config.output_root / "full_train_test" / run_name / "counting",
        }
        for run_name, _ in VARIANTS
    }
    for roots in counting_roots.values():
        for root in roots.values():
            if not (root / "counting_manifest.json").is_file():
                raise FileNotFoundError(root / "counting_manifest.json")
    summary = {
        "pipeline_complete": True,
        "stage": config.stage,
        "experiment_group": config.experiment_group,
        "config": config.to_dict(),
        "all_runs_summary": portable_path(all_runs),
        "statistical_tables": {key: portable_path(path) for key, path in statistical_tables.items()},
        "claims": {key: portable_path(path) for key, path in claims_outputs.items()},
        "oof_aggregation_manifest": portable_path(aggregation_manifest),
        "full_run_manifest": portable_path(full_run_manifest),
        "counting": stringify_path_tree(counting_roots),
    }
    output = config.output_root / "pipeline_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"all_runs_summary": portable_path(all_runs), "pipeline_summary": portable_path(output)}


def run_all(config: RuntimeConfig) -> None:
    validate_data_files(config)
    tokenizer = build_tokenizer(config.model_name)
    for fold_id in range(config.n_folds):
        run_fold(config, fold_id, tokenizer)
    aggregate_results = run_aggregate(config)
    full_results = run_full(config, tokenizer)
    claims_results = run_claims(config)
    summary_results = run_summary(config)
    print(
        json.dumps(
            {
                "cross_validation": aggregate_results,
                "full_train_test": full_results,
                "claims": claims_results,
                "summary": summary_results,
            },
            indent=2,
        )
    )


def main() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "This pipeline requires Python 3.10+ for Transformers 5; Python 3.12 is recommended."
        )
    args = parse_args()
    config = runtime_config(args)
    config.output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "fold":
        if args.fold_index is None:
            raise ValueError("fold command requires --fold-index")
        if not 0 <= args.fold_index < config.n_folds:
            raise ValueError(f"fold index must be in 0..{config.n_folds - 1}")
    assert_command_matches_stage(args.command, config.stage)
    if args.command in {"fold", "full", "all", "seed", "multiseed"}:
        if not (config.train_data_path.is_file() and config.test_data_path.is_file()):
            build_training_npzs(args.raw_dir.expanduser().absolute(), config.data_root)
        validate_data_files(config)
        if not torch.cuda.is_available():
            raise RuntimeError("Training commands require one CUDA GPU")
    if args.command == "fold":
        assert args.fold_index is not None
        run_fold(config, args.fold_index)
    elif args.command == "aggregate":
        run_aggregate(config)
    elif args.command == "full":
        run_full(config)
    elif args.command == "claims":
        run_claims(config)
    elif args.command == "summary":
        run_summary(config)
    elif args.command == "all":
        run_all(config)
    elif args.command == "seed":
        run_one_seed(config, args)
    elif args.command == "multiseed":
        print(json.dumps(run_multiseed(config, requested_seeds(args))["seeds"], indent=2))
    elif args.command == "multiseed-aggregate":
        print(json.dumps(run_multiseed_aggregate(config, args), indent=2))
    elif args.command == "ceiling-report":
        print(json.dumps(run_ceiling_report(config, args), indent=2))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
