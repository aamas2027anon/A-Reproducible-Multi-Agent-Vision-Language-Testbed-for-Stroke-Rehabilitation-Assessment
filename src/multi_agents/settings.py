"""Paths and every tunable threshold of the multi-agent system.

All MAS behaviour that could change a result lives in `MASConfig`, so one hash
identifies a configuration; its defaults are `mas.agents` in config.yaml. Tune it on the
validation subjects, never on test.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

# Settings come from config.yaml through train/config.py, which also defines the paths.
TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
import config as _layout  # noqa: E402  (train/config.py)

REPO_ROOT = _layout.PROJECT_ROOT
RAW_DATA_DIR = _layout.RAW_DATA_DIR
OUTPUT_DIR = _layout.MAS_OUTPUT_DIR
CACHE_DIR = _layout.CACHE_DIR / "mas"
LOG_DIR = _layout.LOG_DIR / "mas"
DEMO_OUTPUT_DIR = _layout.DEMO_OUTPUT_DIR
DEFAULT_SQLITE_PATH = DEMO_OUTPUT_DIR / "memory.sqlite"
DEFAULT_CHECKPOINT = _layout.project_path("checkpoint") if _layout.CONFIG["paths"].get("checkpoint") else None

# The LoRA checkpoint was trained on (and must be loaded onto) the Base model.
BASE_MODEL_REPO = _layout.MODEL_NAME
BASE_MODEL_DIR = _layout.BASE_MODEL_DIR

_MAS = _layout.CONFIG["mas"]
_AGENTS = _MAS["agents"]
STREAM_FPS = float(_MAS["stream_fps"])
TUNING = _MAS["tuning"]

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
    """Agent settings; the meaning of each field is documented under `mas.agents` in config.yaml."""

    # Planning
    transition_strength: float = float(_AGENTS["transition_strength"])
    emission_temperature: float = float(_AGENTS["emission_temperature"])
    population_prior_strength: float = float(_AGENTS["population_prior_strength"])
    patient_count_min_confidence: float = float(_AGENTS["patient_count_min_confidence"])
    # Evaluation
    ue_calib_a: float = float(_AGENTS["ue_calib_a"])
    ue_calib_b: float = float(_AGENTS["ue_calib_b"])
    severe_below: float = float(_AGENTS["severe_below"])
    mild_from: float = float(_AGENTS["mild_from"])
    uefma_between_exercise_sd: float = float(_AGENTS["uefma_between_exercise_sd"])
    uefma_window_decorrelation: float = float(_AGENTS["uefma_window_decorrelation"])
    uefma_drift_sigma: float = float(_AGENTS["uefma_drift_sigma"])
    uefma_min_exercises_for_drift: int = int(_AGENTS["uefma_min_exercises_for_drift"])
    # Safety
    safety_enabled: bool = bool(_AGENTS["safety_enabled"])
    rest_cost: float = float(_AGENTS["rest_cost"])
    escalate_confidence: dict[str, float] = field(
        default_factory=lambda: {band: float(value) for band, value in _AGENTS["escalate_confidence"].items()}
    )
    escalation_rate_limit: float = float(_AGENTS["escalation_rate_limit"])
    escalation_history: int = int(_AGENTS["escalation_history"])
    implausible_transition: float = float(_AGENTS["implausible_transition"])
    # Coordinator
    exercise_escalate_fraction: float = float(_AGENTS["exercise_escalate_fraction"])

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
