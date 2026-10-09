"""Assembly shared by the demo and the evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agents.agents import (
    CoordinatorAgent,
    EvaluationAgent,
    ModelBackend,
    PerceptionAgent,
    PlanningAgent,
    SafetyAgent,
)
from multi_agents.data import ExerciseSource
from multi_agents.graph import Services
from multi_agents.inference import (
    InferenceCache,
    MovementModelRunner,
    checkpoint_identity,
    find_checkpoint,
    load_patient_normalizer,
    read_checkpoint_conditioning,
)
from multi_agents.memory import WindowLog
from multi_agents.model_download import ensure_model, warn_if_not_base_model
from multi_agents.priors import PopulationPriors
from multi_agents.settings import (
    BASE_MODEL_DIR,
    BASE_MODEL_REPO,
    CACHE_DIR,
    MASConfig,
)


@dataclass(frozen=True)
class ModelSetup:
    checkpoint: Path
    base_model: Path
    conditioning: str
    normalizer: Any
    identity: str
    run_name: str | None


def prepare_model(checkpoint: str | None, base_model: str | None, allow_download: bool) -> ModelSetup:
    """Resolve the checkpoint and make sure the Base model it was trained on exists."""
    base = ensure_model(BASE_MODEL_REPO, Path(base_model) if base_model else BASE_MODEL_DIR, allow_download)
    warn_if_not_base_model(base)
    checkpoint_path = find_checkpoint(checkpoint)
    conditioning, hparams = read_checkpoint_conditioning(checkpoint_path)
    return ModelSetup(
        checkpoint=checkpoint_path,
        base_model=base,
        conditioning=conditioning,
        normalizer=load_patient_normalizer(checkpoint_path, conditioning),
        identity=checkpoint_identity(checkpoint_path, base),
        run_name=hparams.get("run_name"),
    )


def make_backend(
    source: ExerciseSource, setup: ModelSetup, device: str | None, batch_size: int, use_cache: bool = True
) -> ModelBackend:
    cache = InferenceCache(CACHE_DIR / "inference", setup.identity) if use_cache else None
    return ModelBackend(
        source,
        lambda: MovementModelRunner(setup.checkpoint, setup.base_model, device=device, batch_size=batch_size),
        cache,
    )


def build_services(
    *,
    config: MASConfig,
    priors: PopulationPriors,
    backend: ModelBackend,
    log: WindowLog,
    mode: str,
) -> Services:
    safety = SafetyAgent(config)
    return Services(
        config=config,
        perception=PerceptionAgent(backend),
        planning=PlanningAgent(priors, config),
        safety=safety,
        evaluation=EvaluationAgent(config),
        coordinator=CoordinatorAgent(config, safety),
        log=log,
        mode=mode,
    )


def patient_seed(source: ExerciseSource, subject_id: str, config: MASConfig, setup: ModelSetup) -> dict[str, Any]:
    return {
        "profile": source.profile(subject_id).model_dump(),
        "config_hash": config.config_hash(),
        "checkpoint": setup.identity,
    }


def log_namespace(config: MASConfig, setup: ModelSetup) -> str:
    return f"{setup.identity}:{config.config_hash()}"
