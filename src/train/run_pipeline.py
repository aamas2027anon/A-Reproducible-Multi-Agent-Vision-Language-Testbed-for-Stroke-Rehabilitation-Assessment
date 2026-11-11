#!/usr/bin/env python3
"""Train the three conditioning variants (none, text, prefix) and write the report.

    python train/run_pipeline.py smoke                                # a few batches, one seed
    python train/run_pipeline.py multiseed                            # all seeds, then the report
    python train/run_pipeline.py multiseed --seed-index 0 --epochs 2  # one seed, chosen epochs
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

# These must be set before importing torch. The first keeps eager operations on the
# standard ATen path instead of Triton JIT, which requires Python.h at runtime; the
# second lets the CUDA allocator reuse fragmented memory; the third keeps the Rust
# tokenizer single-threaded so forked data-loader workers cannot deadlock.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from config import (
    EXPERIMENT_DIRNAME,
    LOG_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PER_DEVICE_BATCH_SIZE,
    RAW_DATA_DIR,
    SEEDS,
    SMOKE_BATCHES,
    TRAINING_DATA_DIR,
    TRAINING_MEMMAP_DIR,
    TRAINING_OUTPUT_DIR,
    TRAINING_RUN_DIR,
    RuntimeConfig,
    portable_path,
)
from data import validate_data_files
from multiseed import run_multiseed, run_seed, seed_from_index
from raw_data import build_training_npzs
from report import export_report
from run_log import start_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("smoke", "multiseed"))
    parser.add_argument(
        "--seed-index",
        type=int,
        help=f"multiseed: train only config.SEEDS[index] (0-{len(SEEDS) - 1}) and write its descriptive report",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="multiseed: maximum epochs per variant (early stopping can end a run sooner)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=TRAINING_DATA_DIR,
        help="Merged train/test NPZs, built here from --raw-dir if missing (default: under paths.output_dir in config.yaml)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Raw StrokeRehab tree (video_extracted_features/, kinematics/); default: paths.data_dir in config.yaml",
    )
    parser.add_argument("--cache-root", type=Path, default=TRAINING_MEMMAP_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"Checkpoints, predictions and the report (default: training/{EXPERIMENT_DIRNAME}/ or training/smoke/ under output_dir)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=PER_DEVICE_BATCH_SIZE,
        help="Windows per device; changing it changes the training recipe",
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = TRAINING_OUTPUT_DIR / "smoke" if args.command == "smoke" else TRAINING_RUN_DIR
    return args


def runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig(
        data_root=args.data_root.expanduser().absolute(),
        output_root=args.output_root.expanduser().absolute(),
        cache_root=args.cache_root.expanduser().absolute(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        resume=not args.no_resume,
    )
    if args.command == "smoke":
        config = dataclasses.replace(config, epochs=1, limit_batches=SMOKE_BATCHES, clip_percentage=100.0)
    if config.batch_size <= 0 or config.num_workers < 0 or config.epochs <= 0:
        raise ValueError("batch size and epochs must be positive and workers cannot be negative")
    return config


def main() -> None:
    args = parse_args()
    config = runtime_config(args)
    name = args.command if args.seed_index is None else f"{args.command}_seed-index-{args.seed_index}"
    start_log(LOG_DIR / "training", name)
    if not (config.train_data_path.is_file() and config.test_data_path.is_file()):
        build_training_npzs(args.raw_dir.expanduser().absolute(), config.data_root)
    validate_data_files(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires one CUDA GPU")

    if args.command == "smoke":
        run_seed(config, SEEDS[0])
        print(f"Smoke run complete: {portable_path(config.output_root)}")
    elif args.seed_index is not None or len(SEEDS) == 1:
        seed = seed_from_index(args.seed_index) if args.seed_index is not None else SEEDS[0]
        run_seed(config, seed)
        print(json.dumps({k: portable_path(v) for k, v in export_report(config, seeds=(seed,)).items()}, indent=2))
    else:
        print(json.dumps(run_multiseed(config)["seeds"]))
        print(json.dumps({k: portable_path(v) for k, v in export_report(config).items()}, indent=2))


if __name__ == "__main__":
    main()
