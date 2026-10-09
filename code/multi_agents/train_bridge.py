"""Single import point for code shared with the training pipeline.

`train/` uses flat imports (`from config import ...`), so its directory is put on
`sys.path` here once. Windowing, prompts, patient encoding, the model definition and
the evaluation metrics all come from `train/`, which keeps the multi-agent system
numerically aligned with training. `training.py` is deliberately not imported: it
pulls in Lightning, which inference does not need.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

from multi_agents.settings import HORIZONS, PRIMITIVES, TRAIN_DIR

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import config as train_config  # noqa: E402
from data import (  # noqa: E402
    PATIENT_NPZ_KEYS,
    PatientNormalizer,
    StrokeRehabCollator,
    StrokeRehabDataset,
    canonical_subject_id,
    extract_npz_members,
)
from raw_data import (  # noqa: E402
    RawClip,
    discover_raw_clips,
    parse_clip_id,
    subjects_by_set,
    write_npz,
)

if tuple(train_config.CLASS_NAMES) != PRIMITIVES:
    raise AssertionError(f"Primitive vocabulary drifted from train/config.py: {train_config.CLASS_NAMES}")
if train_config.TARGET_WINDOW_SIZE != HORIZONS:
    raise AssertionError("Horizon count drifted from train/config.py")


def build_model(model_name: str, use_patient_prefix: bool):
    return _module("model").build_model(model_name, use_patient_prefix)


def classification_summary(frame):
    return _module("artifacts").classification_summary(frame)


def count_video_primitives(frame):
    return _module("counting").count_video_primitives(frame)


def build_count_aggregations(video_counts):
    return _module("counting").build_count_aggregations(video_counts)


def write_final_artifacts(frame, run_dir, run_name: str, split: str):
    """Training's per-run `final/` tables, confusion matrices and counting."""
    return _module("artifacts").write_final_artifacts(frame, run_dir, run_name, split)


def per_class_summary(frame):
    return _module("artifacts").per_class_summary(frame)


def cval_statistics() -> ModuleType:
    """Training's paired estimators (bootstrap CI, sign flip, Cohen's d, Wilcoxon, ...)."""
    return _module("cval_statistics")


def monte_carlo_sign_flip_pvalue(differences, rng):
    return _module("cval_statistics").monte_carlo_sign_flip_pvalue(differences, rng)


def bootstrap_mean_ci(differences, rng):
    return _module("cval_statistics").bootstrap_mean_ci(differences, rng)


def _module(name: str) -> ModuleType:
    """Import heavier training modules (peft, sklearn, matplotlib) only when used."""
    return importlib.import_module(name)


__all__ = [
    "PATIENT_NPZ_KEYS",
    "PatientNormalizer",
    "RawClip",
    "StrokeRehabCollator",
    "StrokeRehabDataset",
    "bootstrap_mean_ci",
    "build_count_aggregations",
    "build_model",
    "canonical_subject_id",
    "classification_summary",
    "count_video_primitives",
    "cval_statistics",
    "discover_raw_clips",
    "extract_npz_members",
    "monte_carlo_sign_flip_pvalue",
    "parse_clip_id",
    "per_class_summary",
    "subjects_by_set",
    "train_config",
    "write_final_artifacts",
    "write_npz",
]
