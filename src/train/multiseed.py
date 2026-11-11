"""Multi-seed training: every seed trains all three conditioning variants and predicts
the test subjects.

Every variant at a seed is handed the same seed value, so the base-vs-conditioning
contrasts are paired by seed.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

from config import (
    FULL_VALIDATION_SUBJECTS,
    SEEDS,
    TRAINING_VARIANTS,
    VARIANTS,
    RuntimeConfig,
    portable_path,
)


SEED_MANIFEST_NAME = "seed_manifest.json"


def seeds_root(config: RuntimeConfig) -> Path:
    return config.output_root / "seeds"


def seed_dir(config: RuntimeConfig, seed: int) -> Path:
    return seeds_root(config) / f"seed_{seed}"


def seed_run_dir(config: RuntimeConfig, seed: int, run_name: str) -> Path:
    return seed_dir(config, seed) / run_name


def seed_from_index(seed_index: int) -> int:
    if not 0 <= seed_index < len(SEEDS):
        raise ValueError(f"seed index must be in 0..{len(SEEDS) - 1}")
    return SEEDS[seed_index]


def seed_is_complete(
    manifest: Mapping[str, Any], seed: int, requested_epochs: int,
    clip_percentage: float = 100.0,
) -> bool:
    """Whether a stored seed manifest already covers the requested epoch count.

    A seed trained to fewer epochs than are now asked for must be extended, not skipped,
    which is why this compares epochs rather than only checking the completion flag.
    """
    if not manifest.get("seed_complete") or manifest.get("seed") != seed:
        return False
    if float(manifest.get("clip_percentage", 100.0)) != clip_percentage:
        return False
    return int(manifest.get("epochs") or 0) >= requested_epochs


def run_seed(config: RuntimeConfig, seed: int, tokenizer: Any | None = None) -> dict[str, Any]:
    """Train and predict all three variants at one seed."""
    from data import npz_subjects
    from training import build_tokenizer, free_gpu_memory, run_experiment

    root = seed_dir(config, seed)
    manifest_path = root / SEED_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_epochs = int(manifest.get("epochs") or 0)
        if seed_is_complete(manifest, seed, config.epochs, config.clip_percentage):
            print(f"Seed {seed} is already complete at {recorded_epochs} epochs; skipping")
            return manifest
        if (manifest.get("seed_complete") and manifest.get("seed") == seed
                and float(manifest.get("clip_percentage", 100.0)) == config.clip_percentage):
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
    for run_name, conditioning in TRAINING_VARIANTS:
        print(
            f"\nSeed {seed}: {run_name} ({conditioning}) | train={len(train_subjects)} "
            f"validation={len(validation_subjects)} test={len(test_subjects)}"
        )
        try:
            results[run_name] = run_experiment(
                config=seed_config,
                tokenizer=tokenizer,
                run_dir=seed_run_dir(seed_config, seed, run_name),
                run_name=run_name,
                conditioning=conditioning,
                train_npz=seed_config.train_data_path,
                train_subjects=train_subjects,
                validation_subjects=validation_subjects,
                prediction_npz=seed_config.test_data_path,
                prediction_subjects=test_subjects,
            )
        finally:
            # Release before the next variant builds its model: the full-batch preflight
            # is the largest allocation in the pipeline and has very little headroom on
            # a 40 GB card, so anything still resident here makes the next variant fail.
            free_gpu_memory(f"variant {run_name}")
        recorded_seed = results[run_name]["training_manifest"]["seed"]
        if int(recorded_seed) != int(seed):
            raise AssertionError(
                f"{run_name} trained at seed {recorded_seed}, expected {seed}; "
                "the variants would not be paired"
            )

    manifest = {
        "seed_complete": True,
        "seed": seed,
        "clip_percentage": config.clip_percentage,
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
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_multiseed(config: RuntimeConfig) -> dict[str, Any]:
    """Train every seed serially on the one GPU."""
    from training import build_tokenizer

    tokenizer = build_tokenizer(config.model_name)
    manifests = {str(seed): run_seed(config, seed, tokenizer) for seed in SEEDS}
    summary = {
        "multiseed_complete": True,
        "seeds": list(SEEDS),
        "variants": [run_name for run_name, _ in VARIANTS],
        "config": config.to_dict(),
        "seed_manifests": {
            str(seed): portable_path(seed_dir(config, seed) / SEED_MANIFEST_NAME)
            for seed in SEEDS
        },
    }
    output = config.output_root / "multiseed_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": portable_path(output), "seeds": list(SEEDS), "manifests": manifests}
