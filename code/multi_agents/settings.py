"""Paths and every tunable threshold of the multi-agent system.

All MAS behaviour that could change a result lives in `MASConfig`, so one hash
identifies a configuration. Tune it on the validation subjects, never on test.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

# Paths come from project_config.toml at the repository root, or from the file named
# by STROKE_REHAB_CONFIG, so the same code runs unchanged on another machine.
_DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "project_config.toml"
CONFIG_FILE = Path(os.environ.get("STROKE_REHAB_CONFIG", _DEFAULT_CONFIG_FILE)).expanduser().resolve()
_PROJECT = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.is_file() else {}
_PATHS = _PROJECT.get("paths", {})
_MODELS = _PROJECT.get("models", {})
_STREAM = _PROJECT.get("stream", {})

_root = Path(_PATHS.get("project_root", ".")).expanduser()
REPO_ROOT = (_root if _root.is_absolute() else CONFIG_FILE.parent / _root).resolve()


def _path(key: str, default: str) -> Path:
    value = Path(str(_PATHS.get(key, default))).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _optional_path(key: str) -> Path | None:
    return _path(key, "") if _PATHS.get(key) else None


TRAIN_DIR = _path("train_dir", "train")
RAW_DATA_DIR = _path("data_dir", "data/stroke_rehab")
MODELS_DIR = _path("models_dir", "models")
CHECKPOINT_DIR = _path("checkpoints_dir", "models/checkpoints")
DEFAULT_CHECKPOINT = _optional_path("checkpoint")
OUTPUT_DIR = _path("output_dir", "multi_agent_outputs")
CACHE_DIR = _path("cache_dir", "multi_agent_outputs/cache")
DEFAULT_SQLITE_PATH = _path("memory_path", "multi_agent_outputs/memory.sqlite")

# The LoRA checkpoint was trained on (and must be loaded onto) the Base model.
BASE_MODEL_REPO = _MODELS.get("base_model_repo", "Qwen/Qwen3.5-0.8B-Base")
BASE_MODEL_DIR = _path("base_model_dir", "models/qwen3.5-0.8B-Base")

# Frames per second of the feature stream (real-time pacing and latency budget).
STREAM_FPS = float(_STREAM.get("fps", 15.0))

PRIMITIVES = ("reach", "reposition", "rest", "stabilize", "transport")
NUM_PRIMITIVES = len(PRIMITIVES)
HORIZONS = 5
REST = "rest"
REST_ID = PRIMITIVES.index(REST)
ACTIVE_PRIMITIVES = ("reach", "reposition", "transport")
ACTIVE_IDS = tuple(PRIMITIVES.index(name) for name in ACTIVE_PRIMITIVES)
BANDS = ("severe", "moderate", "mild")
UEFMA_MAX = 66.0

VERDICTS = ("approve", "revise", "escalate", "block")


@dataclass(frozen=True)
class MASConfig:
    # Planning: online forward filter over the VLM's per-frame posteriors,
    #   belief_h ∝ q_h ** temperature * predicted_h ** transition_strength
    # With transition_strength = 0 and temperature = 1 the plan is
    # exactly the model's own prediction ("do no harm"); tune.py raises them only where
    # the validation subjects show a gain.
    transition_strength: float = 0.0
    emission_temperature: float = 1.0
    # Pseudo-counts per transition row (~4 exercises of patient history to reach 50/50).
    population_prior_strength: float = 2000.0
    patient_count_min_confidence: float = 0.6

    # Evaluation: UE-FMA bands on the MAS's own estimate (never the ground truth).
    # Pooling (mean, variance, weights, drift) runs on the model's raw scale; every
    # reported estimate and the band use the linear calibration
    # mu_cal = clip(ue_calib_a + ue_calib_b * mu, 0, 66), sigma_cal = ue_calib_b * sigma,
    # fitted on the validation subjects by tune.py. The defaults leave scores unchanged.
    ue_calib_a: float = 0.0
    ue_calib_b: float = 1.0
    severe_below: float = 26.0
    mild_from: float = 53.0
    uefma_between_exercise_sd: float = 4.0
    uefma_window_decorrelation: float = 23.0
    uefma_drift_sigma: float = 2.5
    uefma_min_exercises_for_drift: int = 3

    # Safety. `safety_enabled=False` is the no-safety ablation (only R1, unusable input,
    # still blocks). Per window: R2 withholds the cue when b_0(plan_0) is below
    # `escalate_confidence[band]` (stricter as the estimated impairment gets worse);
    # otherwise C1 cues, at each frame, the a maximizing
    # P(a) - rest_cost * P(rest) * [a is a movement]. The coordinator arbitrates
    # perception/planning disagreements with the same cost. `rest_cost` is tuned on
    # validation (tune.py).
    safety_enabled: bool = True
    rest_cost: float = 0.5
    escalate_confidence: dict[str, float] = field(
        default_factory=lambda: {"severe": 0.35, "moderate": 0.30, "mild": 0.25}
    )
    # Exercise rules E1 (escalation rate) and E2 (UE-FMA drift, see the evaluation settings).
    escalation_rate_limit: float = 0.30
    escalation_history: int = 3
    # Metric only: population transitions below this probability count as implausible.
    implausible_transition: float = 1e-3

    # Coordinator.
    exercise_escalate_fraction: float = 0.25

    def __post_init__(self) -> None:
        # C1 never changing a rest/stabilize cue needs rest_cost >= 0; sigma_cal = b * sigma needs b > 0.
        if self.rest_cost < 0:
            raise ValueError(f"rest_cost must be >= 0, got {self.rest_cost}")
        if self.ue_calib_b <= 0:
            raise ValueError(f"ue_calib_b must be > 0, got {self.ue_calib_b}")

    def calibrate_uefma(self, score):
        """mu_cal = clip(a + b * score, 0, 66) for a score or an array of scores (NaN and None pass through)."""
        if score is None:
            return None
        calibrated = np.clip(self.ue_calib_a + self.ue_calib_b * np.asarray(score, dtype=np.float64), 0.0, UEFMA_MAX)
        return float(calibrated) if calibrated.ndim == 0 else calibrated

    def band(self, uefma: float | None) -> str:
        """Impairment band of a calibrated UE-FMA estimate."""
        if uefma is None:
            return "moderate"
        if uefma < self.severe_below:
            return "severe"
        if uefma >= self.mild_from:
            return "mild"
        return "moderate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    @classmethod
    def from_json(cls, path: str | Path | None) -> "MASConfig":
        base = cls()
        if path is None:
            return base
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        # Unknown keys (settings since removed) are ignored, so older tuned configs still load.
        known = set(base.to_dict())
        unknown = sorted(set(overrides) - known)
        if unknown:
            warnings.warn(f"Ignoring unknown MASConfig keys in {path}: {unknown}", stacklevel=2)
        return replace(base, **{key: value for key, value in overrides.items() if key in known})
