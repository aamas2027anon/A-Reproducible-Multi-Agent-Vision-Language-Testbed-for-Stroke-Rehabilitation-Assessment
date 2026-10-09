"""Multi-seed replication of the full train/test stage.

Every seed trains all three conditioning variants at the same seed value, so the
base-vs-conditioning contrasts are paired by seed. The cross-validation fold logic is
not involved here.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from artifacts import write_final_artifacts
from config import (
    BASE_RUN_NAME,
    FULL_VALIDATION_SUBJECTS,
    N_SEEDS,
    RuntimeConfig,
    SEED,
    SEEDS,
    TRAINING_VARIANTS,
    VARIANTS,
    portable_path,
)
from counting import stringify_path_tree
from data import npz_subjects
from epoch_metrics import epoch_metrics_callback_factory, release_epoch_metrics_callbacks
from training import build_tokenizer, free_gpu_memory, run_experiment


SEED_MANIFEST_NAME = "seed_manifest.json"
RESOLVED_SEEDS_NAME = "resolved_seeds.json"
COUNT_TRUTH_COLUMNS = (
    "subject_id",
    "task_name",
    "clip_id",
    "label_id",
    "primitive",
    "actual_count",
)


# --------------------------------------------------------------------------------
# Seed pool
# --------------------------------------------------------------------------------


def resolve_seeds(
    requested: Iterable[int],
    n_seeds: int = N_SEEDS,
    base_seed: int = SEED,
) -> tuple[tuple[int, ...], list[str]]:
    """De-duplicate and top up a seed list. Never raises on a malformed request.

    A queued job must not die because the configured list repeats a value or is short,
    so duplicates are dropped and replacements are drawn deterministically from
    `base_seed`. Returns the resolved seeds and a human-readable list of the repairs.
    """
    if n_seeds <= 0:
        raise ValueError(f"n_seeds must be positive; got {n_seeds}")
    repairs: list[str] = []
    resolved: list[int] = []
    for value in requested:
        seed = int(value)
        if seed in resolved:
            repairs.append(f"seed {seed} appeared more than once; dropped the repeat")
            continue
        resolved.append(seed)
    if len(resolved) > n_seeds:
        repairs.append(
            f"received {len(resolved)} seeds but only {n_seeds} are used; "
            f"dropped {resolved[n_seeds:]}"
        )
        resolved = resolved[:n_seeds]
    generator = np.random.default_rng(base_seed)
    while len(resolved) < n_seeds:
        candidate = int(generator.integers(1, 2**31 - 1))
        if candidate in resolved:
            continue
        resolved.append(candidate)
        repairs.append(f"seed list was short; added {candidate}")
    for line in repairs:
        print(f"Seed resolution: {line}")
    return tuple(resolved), repairs


def seeds_root(config: RuntimeConfig) -> Path:
    return config.output_root / "seeds"


def load_or_create_resolved_seeds(
    config: RuntimeConfig, requested: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Resolve the seed pool once per output root, then always re-read that file.

    Persisting the pool is what makes `--seed-index N` stable across submissions and
    guarantees the aggregation step sees exactly the seeds that were trained.
    """
    path = seeds_root(config) / RESOLVED_SEEDS_NAME
    stage_seeds = config.stage_spec.seeds or SEEDS
    requested_seeds = tuple(
        int(value) for value in (requested if requested is not None else stage_seeds)
    )
    # The stage decides how many seeds the sweep has; --seeds only supplies candidate
    # values, which resolve_seeds then de-duplicates and tops up to this length. Taking
    # the size from the request instead would silently shrink a ten-seed sweep to two
    # when someone passed a duplicate, and would top the single-seed ceiling stage up
    # to ten.
    pool_size = len(stage_seeds)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        existing = tuple(int(value) for value in stored["resolved"])
        if requested is not None and tuple(stored.get("requested", ())) != requested_seeds:
            print(
                f"Seed resolution: {path} already pins {list(existing)}; "
                f"ignoring the newly requested {list(requested_seeds)}"
            )
        return existing
    resolved, repairs = resolve_seeds(requested_seeds, n_seeds=pool_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage": config.stage,
                "requested": list(requested_seeds),
                "resolved": list(resolved),
                "repairs": repairs,
                "n_seeds": len(resolved),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return resolved


def seed_from_index(config: RuntimeConfig, seed_index: int) -> int:
    resolved = load_or_create_resolved_seeds(config)
    if not 0 <= seed_index < len(resolved):
        raise ValueError(f"seed index must be in 0..{len(resolved) - 1}")
    return resolved[seed_index]


def seed_dir(config: RuntimeConfig, seed: int) -> Path:
    return seeds_root(config) / f"seed_{seed}"


def seed_run_dir(config: RuntimeConfig, seed: int, run_name: str) -> Path:
    return seed_dir(config, seed) / run_name


# --------------------------------------------------------------------------------
# Final (end-of-run) per-seed artifacts
# --------------------------------------------------------------------------------


def subject_total_counts_path(run_dir: Path, split: str = "test") -> Path:
    """Where `write_final_artifacts` leaves the per-subject total counts."""
    return run_dir / "final" / "counting" / "predictions" / f"{split}_subject_total_counts.csv"


def task_metrics_path(run_dir: Path) -> Path:
    return run_dir / "final" / "tables" / "task_metrics.csv"


# --------------------------------------------------------------------------------
# One seed
# --------------------------------------------------------------------------------


def _assert_aligned_counts(counting_results: dict[str, dict[str, Any]]) -> None:
    """Every variant must have scored the identical ground-truth counts."""
    truth_columns = list(COUNT_TRUTH_COLUMNS)
    sort_keys = truth_columns[:-1]
    reference = None
    for run_name, _ in VARIANTS:
        counts = pd.read_csv(counting_results[run_name]["predictions"]["video_primitive"])
        truth = (
            counts[truth_columns].sort_values(sort_keys).reset_index(drop=True)
        )
        if reference is None:
            reference = truth
            continue
        pd.testing.assert_frame_equal(
            truth,
            reference,
            check_dtype=False,
            obj=f"{run_name} vs {BASE_RUN_NAME} actual primitive counts",
        )


def seed_is_complete(
    manifest: Mapping[str, Any], seed: int, requested_epochs: int
) -> bool:
    """Whether a stored seed manifest already covers the requested epoch count.

    A seed trained to fewer epochs than are now asked for must be extended, not skipped,
    which is why this compares epochs rather than only checking the completion flag.
    """
    if not manifest.get("seed_complete") or manifest.get("seed") != seed:
        return False
    return int(manifest.get("epochs") or 0) >= requested_epochs


def run_seed(config: RuntimeConfig, seed: int, tokenizer: Any | None = None) -> dict[str, Any]:
    """Train and evaluate all three variants at one seed.

    The variant loop lives inside this function on purpose: every variant is handed the
    same seed integer, so they share the loader shuffle order and the comparison differs
    only in conditioning.
    """
    root = seed_dir(config, seed)
    manifest_path = root / SEED_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_epochs = int(manifest.get("epochs") or 0)
        if seed_is_complete(manifest, seed, config.epochs):
            print(f"Seed {seed} is already complete at {recorded_epochs} epochs; skipping")
            return manifest
        if manifest.get("seed_complete") and manifest.get("seed") == seed:
            print(
                f"Seed {seed} completed {recorded_epochs} epochs but {config.epochs} are "
                "requested; extending every variant"
            )

    if tokenizer is None:
        tokenizer = build_tokenizer(config.model_name)
    seed_config = dataclasses.replace(config, seed=seed)

    training_pool = frozenset(npz_subjects(seed_config.train_data_path, seed_config.cache_root))
    missing_validation = FULL_VALIDATION_SUBJECTS - training_pool
    if missing_validation:
        raise ValueError(
            f"Configured validation subjects are missing: {sorted(missing_validation)}"
        )
    validation_subjects = tuple(sorted(FULL_VALIDATION_SUBJECTS))
    train_subjects = tuple(sorted(training_pool - FULL_VALIDATION_SUBJECTS))
    test_subjects = npz_subjects(seed_config.test_data_path, seed_config.cache_root)
    if set(validation_subjects) & set(test_subjects):
        raise AssertionError("Validation and test subjects overlap")

    results: dict[str, Any] = {}
    counting_results: dict[str, dict[str, Any]] = {}
    final_results: dict[str, Any] = {}
    for run_name, conditioning in TRAINING_VARIANTS:
        run_dir = seed_run_dir(seed_config, seed, run_name)
        print(
            f"\nSeed {seed}: {run_name} ({conditioning}) | train={len(train_subjects)} "
            f"validation={len(validation_subjects)} test={len(test_subjects)}"
        )
        factory = epoch_metrics_callback_factory(
            seed_config, run_name, conditioning, run_dir / "epochs"
        )
        try:
            results[run_name] = run_experiment(
                config=seed_config,
                tokenizer=tokenizer,
                run_dir=run_dir,
                run_name=run_name,
                conditioning=conditioning,
                train_npz=seed_config.train_data_path,
                train_subjects=train_subjects,
                validation_subjects=validation_subjects,
                prediction_npz=seed_config.test_data_path,
                prediction_subjects=test_subjects,
                fold_id=-1,
                split="test",
                epoch_metrics_callback_factory=factory,
            )
        finally:
            # Release before the next variant builds its model: the batch-72 preflight
            # is the largest allocation in the pipeline and has very little headroom on
            # a 40 GB card, so anything still resident here makes the next variant fail.
            release_epoch_metrics_callbacks(factory)
            del factory
            free_gpu_memory(f"variant {run_name}")
        recorded_seed = results[run_name]["training_manifest"]["seed"]
        if int(recorded_seed) != int(seed):
            raise AssertionError(
                f"{run_name} trained at seed {recorded_seed}, expected {seed}; "
                "the variants would not be paired"
            )
        combined = pd.read_csv(results[run_name]["predictions"]["combined"])
        final_results[run_name] = write_final_artifacts(
            combined, run_dir, run_name, "test"
        )
        counting_results[run_name] = final_results[run_name]["counting"]

    _assert_aligned_counts(counting_results)
    manifest = {
        "seed_complete": True,
        "seed": seed,
        "stage": config.stage,
        # Guards the skip above: a later run asking for more epochs must extend, not skip.
        "epochs": config.epochs,
        "completed_epochs": min(
            int(result["training_manifest"]["completed_epochs"])
            for result in results.values()
        ),
        "train_subjects": list(train_subjects),
        "validation_subjects": list(validation_subjects),
        "test_subjects": list(test_subjects),
        "variants": results,
        "counting": stringify_path_tree(counting_results),
        "final": final_results,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_multiseed(
    config: RuntimeConfig, requested: Sequence[int] | None = None
) -> dict[str, Any]:
    """Train every resolved seed serially on the one GPU."""
    resolved = load_or_create_resolved_seeds(config, requested)
    tokenizer = build_tokenizer(config.model_name)
    manifests = {}
    for seed in resolved:
        manifests[str(seed)] = run_seed(config, seed, tokenizer)
    summary = {
        "multiseed_complete": True,
        "seeds": list(resolved),
        "variants": [run_name for run_name, _ in VARIANTS],
        "config": config.to_dict(),
        "seed_manifests": {
            str(seed): str(seed_dir(config, seed) / SEED_MANIFEST_NAME)
            for seed in resolved
        },
    }
    output = config.output_root / "multiseed_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": portable_path(output), "seeds": list(resolved), "manifests": manifests}
