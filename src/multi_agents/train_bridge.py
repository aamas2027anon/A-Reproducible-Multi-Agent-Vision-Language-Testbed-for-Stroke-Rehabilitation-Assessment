"""Single import point for code shared with the training pipeline.

`train/` uses flat imports (`from config import ...`), so its directory is put on
`sys.path` here once. Windowing, prompts, patient encoding and the model definition
all come from `train/`, which keeps the multi-agent system numerically aligned with
training. `training.py` is deliberately not imported: it
pulls in Lightning, which inference does not need.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

from multi_agents.settings import HORIZONS, PRIMITIVES, TRAIN_DIR

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
from base_model import ensure_model  # noqa: E402
from run_log import start_log  # noqa: E402
from sign_flip import sign_flip_pvalue  # noqa: E402

if tuple(train_config.CLASS_NAMES) != PRIMITIVES:
    raise AssertionError(f"Primitive vocabulary drifted from train/config.py: {train_config.CLASS_NAMES}")
if train_config.TARGET_WINDOW_SIZE != HORIZONS:
    raise AssertionError("Horizon count drifted from train/config.py")


def build_model(model_name: str, use_patient_prefix: bool):
    return _module("model").build_model(model_name, use_patient_prefix)


def _module(name: str) -> ModuleType:
    """Import the model module (peft, transformers) only when used."""
    return importlib.import_module(name)


__all__ = [
    "PATIENT_NPZ_KEYS",
    "PatientNormalizer",
    "RawClip",
    "StrokeRehabCollator",
    "StrokeRehabDataset",
    "build_model",
    "canonical_subject_id",
    "discover_raw_clips",
    "ensure_model",
    "extract_npz_members",
    "parse_clip_id",
    "start_log",
    "sign_flip_pvalue",
    "subjects_by_set",
    "train_config",
    "write_npz",
]
