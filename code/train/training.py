from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from config import (
    AUTO_RESUME,
    CLASS_NAMES,
    CONDITIONED_VISION_DIM,
    CONDITIONING_MODE,
    CONDITIONING_MODES,
    CONDITIONING_PREFIX,
    CONDITIONING_TEXT,
    CONTEXT_WINDOW_SIZE,
    DETERMINISTIC_TRAINING,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    ENABLE_EARLY_STOPPING,
    FRAME_SKIP,
    FULL_VALIDATION_SUBJECTS,
    GRADIENT_ACCUMULATION_STEPS,
    GRAD_CLIP_NORM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MASK_LABEL_ID,
    MAX_TEXT_LENGTH,
    NUM_EPOCHS,
    NUM_PRIMITIVE_CLASSES,
    N_FOLDS,
    PATIENT_FEATURE_DIM,
    PATIENT_PREFIX_DIM,
    POSITION_NAMES,
    PREDICTION_PROMPT,
    QWEN_HIDDEN_SIZE,
    RuntimeConfig,
    SCHEDULER_GAMMA,
    SOURCE_GESTURE_SEQUENCE_LENGTH,
    TARGET_FRAME_OFFSET,
    TARGET_WINDOW_SIZE,
    TASK_DESCRIPTIONS,
    UEFMA_LOSS_WEIGHT,
    UEFMA_MAX,
    UEFMA_MIN,
    USE_GRADIENT_CHECKPOINTING,
    VISION_FEATURE_DIM,
    WEIGHT_DECAY,
    WINDOW_STRIDE,
    portable_path,
)
from data import (
    PatientNormalizer,
    StrokeRehabCollator,
    StrokeRehabDataset,
    make_loader,
)
from model import build_model, compute_joint_loss, model_batch


PRIMITIVE_EXPORT_COLUMNS = (
    "subject_id",
    "clip_id",
    "task_name",
    "target_frame_index",
    "horizon",
    "actual_primitive",
    "predicted_primitive",
    "prediction_probability",
    "run_name",
    "fold_id",
    "split",
)
UEFMA_EXPORT_COLUMNS = (
    "subject_id",
    "actual_ue_fma",
    "predicted_ue_fma",
    "absolute_error",
    "run_name",
    "fold_id",
    "split",
)


def reset_run_seed(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = DETERMINISTIC_TRAINING
    torch.use_deterministic_algorithms(
        DETERMINISTIC_TRAINING, warn_only=DETERMINISTIC_TRAINING
    )


def free_gpu_memory(label: str) -> dict[str, float]:
    """Drop cached CUDA blocks and report what is still live.

    The full-batch preflight is the largest single allocation in the pipeline and runs
    within a couple of GiB of a 40 GB card, so anything a previous variant leaves
    resident is enough to make the next one fail. Printing allocated-vs-reserved at
    each boundary turns that from a guess into a number in the log.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(
        f"GPU memory after {label}: {allocated:.2f} GiB allocated, "
        f"{reserved:.2f} GiB reserved"
    )
    return {"allocated_gib": allocated, "reserved_gib": reserved}


def training_precision() -> str:
    if not torch.cuda.is_available():
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


def build_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer exposes neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    report_prompt_token_headroom(tokenizer)
    return tokenizer


def report_prompt_token_headroom(tokenizer: Any) -> dict[str, int]:
    """Print the longest prompt each conditioning mode can produce.

    The collator truncates on the right, so an over-long prompt would silently drop
    PREDICTION_PROMPT rather than fail. This only reports; the patient-text variant ran
    at MAX_TEXT_LENGTH=256 in the source notebook with these same strings.
    """
    # Longest plausible values for each field, so the reported number is an upper bound.
    worst_case_patient = (
        "Patient information: paretic side is right; gender is Female; "
        "age is 100 years; race is Unknown/Not Reported; time since stroke is 1000.0."
    )
    longest: dict[str, int] = {}
    for mode in CONDITIONING_MODES:
        widest = 0
        for description in TASK_DESCRIPTIONS.values():
            if mode == CONDITIONING_TEXT:
                text = f"{description}\n{worst_case_patient}\n{PREDICTION_PROMPT}"
            else:
                text = f"{description}\n{PREDICTION_PROMPT}"
            widest = max(widest, len(tokenizer(text)["input_ids"]))
        longest[mode] = widest
        status = "truncates" if widest > MAX_TEXT_LENGTH else "fits"
        print(
            f"Prompt headroom [{mode}]: worst case {widest} tokens "
            f"of {MAX_TEXT_LENGTH} ({status})"
        )
    return longest


def _portable_strings(value: Any) -> Any:
    """`value` with every absolute path string rewritten by `portable_path`."""
    if isinstance(value, str):
        return portable_path(value) if os.path.isabs(value) else value
    if isinstance(value, Mapping):
        return {_portable_strings(key): _portable_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_portable_strings(item) for item in value)
    return value


class QwenLightningModule(L.LightningModule):
    """Joint primitive/UE-FMA training and prediction wrapper."""

    def __init__(
        self,
        run_name: str,
        conditioning: str,
        model_name: str,
        learning_rate: float,
        weight_decay: float = WEIGHT_DECAY,
        scheduler_gamma: float = SCHEDULER_GAMMA,
    ) -> None:
        super().__init__()
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"Unknown conditioning {conditioning!r}; expected one of {CONDITIONING_MODES}"
            )
        self.strict_loading = False
        self.save_hyperparameters()
        # Only the prefix variant changes the network; "text" differs from "none" in the
        # prompt alone and therefore builds the identical architecture.
        self.model, self.model_metadata = build_model(
            model_name, conditioning == CONDITIONING_PREFIX
        )
        self._compact_state_key_names = frozenset(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        vision_feat: torch.Tensor,
        patient_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            text_attention_mask,
            vision_feat,
            patient_features,
        )

    def _shared_step(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        logits, predicted_fma = self(**model_batch(batch))
        losses = compute_joint_loss(
            logits,
            predicted_fma,
            batch["gesture_labels"],
            batch["ue_fma_score"],
        )
        return logits, predicted_fma, losses

    @staticmethod
    def _batch_primitive_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predicted = logits.argmax(dim=-1)
        valid = targets != MASK_LABEL_ID
        return (predicted[valid] == targets[valid]).float().mean()

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        logits, _, losses = self._shared_step(batch)
        batch_size = int(batch["gesture_labels"].shape[0])
        self.log_dict(
            {
                "train_joint_loss": losses.total,
                "train_primitive_loss": losses.primitive,
                "train_ue_fma_normalized_mse": losses.ue_fma_normalized,
                "train_accuracy": self._batch_primitive_accuracy(
                    logits, batch["gesture_labels"]
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return losses.total

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        logits, _, losses = self._shared_step(batch)
        batch_size = int(batch["gesture_labels"].shape[0])
        self.log_dict(
            {
                "val_joint_loss": losses.total,
                "val_primitive_loss": losses.primitive,
                "val_ue_fma_normalized_mse": losses.ue_fma_normalized,
                "val_accuracy": self._batch_primitive_accuracy(
                    logits, batch["gesture_labels"]
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return losses.total

    def predict_step(
        self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        logits, predicted_fma = self(**model_batch(batch))
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted_ids = probabilities.argmax(dim=-1)
        predicted_probabilities = probabilities.gather(
            -1, predicted_ids.unsqueeze(-1)
        ).squeeze(-1)
        return {
            "labels": batch["gesture_labels"].detach().cpu(),
            "predicted_ids": predicted_ids.detach().cpu(),
            "predicted_probabilities": predicted_probabilities.detach().cpu(),
            "actual_fma": batch["ue_fma_score"].detach().float().cpu(),
            "predicted_fma": predicted_fma.detach().float().cpu(),
            "subject_id": list(batch["subject_id"]),
            "clip_id": list(batch["clip_id"]),
            "task_name": list(batch["task_name"]),
            "target_frame_index": list(batch["target_frame_index"]),
        }

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.hparams.scheduler_gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        compact = {
            name: value
            for name, value in checkpoint["state_dict"].items()
            if name in self._compact_state_key_names
        }
        missing = self._compact_state_key_names - compact.keys()
        if missing:
            raise KeyError(f"Compact checkpoint missing trainable keys: {sorted(missing)[:10]}")
        checkpoint["state_dict"] = compact
        checkpoint["compact_trainable_state"] = True
        checkpoint["model_metadata"] = self.model_metadata
        # Lightning stores absolute checkpoint paths (ModelCheckpoint dirpath, last and
        # best paths); keep them relative so a shared checkpoint carries no local paths.
        for key in ("callbacks", "hyper_parameters", "model_metadata"):
            if key in checkpoint:
                checkpoint[key] = _portable_strings(checkpoint[key])

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if not checkpoint.get("compact_trainable_state"):
            raise ValueError("Expected a compact Qwen checkpoint")
        missing = self._compact_state_key_names - checkpoint["state_dict"].keys()
        if missing:
            raise KeyError(f"Checkpoint missing trainable keys: {sorted(missing)[:10]}")


def conditioning_mode_label(conditioning: str) -> str:
    if conditioning == CONDITIONING_PREFIX:
        return CONDITIONING_MODE
    if conditioning == CONDITIONING_TEXT:
        return "patient_text_prompt"
    return "none"


def checkpoint_configuration_id(config: RuntimeConfig, conditioning: str) -> str:
    if conditioning not in CONDITIONING_MODES:
        raise ValueError(
            f"Unknown conditioning {conditioning!r}; expected one of {CONDITIONING_MODES}"
        )
    use_patient_prefix = conditioning == CONDITIONING_PREFIX
    payload = {
        "pipeline_schema": 8,
        # Stage-derived, so ceiling / multiseed / crossval never share checkpoints.
        "experiment_group": config.experiment_group,
        "model_name": config.model_name,
        # "conditioning" is what separates the base and text variants: they build the
        # same architecture, so without it their checkpoint namespaces would collide.
        "conditioning": conditioning,
        "use_patient_prefix": bool(use_patient_prefix),
        "conditioning_mode": conditioning_mode_label(conditioning),
        "seed": config.seed,
        "n_folds": config.n_folds,
        "full_validation_subjects": sorted(FULL_VALIDATION_SUBJECTS),
        # "epochs" is deliberately absent: it says how far a run has progressed, not
        # which experiment it is. Keeping it out is what lets a finished run be resumed
        # for more epochs into the same checkpoint directory.
        "learning_rate": config.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "scheduler_gamma": SCHEDULER_GAMMA,
        "batch_size": config.batch_size,
        "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": config.batch_size * GRADIENT_ACCUMULATION_STEPS,
        "temporal_sampling": {
            "context_window_size": CONTEXT_WINDOW_SIZE,
            "frame_skip": FRAME_SKIP,
            "window_stride": WINDOW_STRIDE,
            "target_window_size": TARGET_WINDOW_SIZE,
            "target_frame_offset": TARGET_FRAME_OFFSET,
            "source_gesture_sequence_length": SOURCE_GESTURE_SEQUENCE_LENGTH,
            "position_names": list(POSITION_NAMES),
        },
        "tokenization": {
            "max_text_length": MAX_TEXT_LENGTH,
            "prediction_prompt": PREDICTION_PROMPT,
        },
        "dimensions": {
            "vision_feature": VISION_FEATURE_DIM,
            "qwen_hidden": QWEN_HIDDEN_SIZE,
            "patient_feature": PATIENT_FEATURE_DIM,
            "patient_prefix": PATIENT_PREFIX_DIM,
            "conditioned_visual": CONDITIONED_VISION_DIM,
        },
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "scaling": LORA_ALPHA / LORA_R,
            "targets": list(LORA_TARGET_MODULES),
        },
        "joint_loss": {
            "primitive": "cross_entropy",
            "ue_fma": "normalized_mse",
            "ue_fma_weight": UEFMA_LOSS_WEIGHT,
        },
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
        "early_stopping": {
            "enabled": ENABLE_EARLY_STOPPING,
            "patience": EARLY_STOPPING_PATIENCE,
            "min_delta": EARLY_STOPPING_MIN_DELTA,
        },
        "output_architecture": {
            "mode": "parallel_primitive_and_uefma_queries",
            "primitive_queries": TARGET_WINDOW_SIZE,
            "uefma_queries": 1,
        },
        "ue_fma_path": "dedicated_fma_query_1024_to_score",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"qwen35-0p8b-parallel-{conditioning}-{digest}"


def _runtime_environment() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_gib": properties.total_memory / (1024**3),
        }
    packages = {}
    for name in ("lightning", "transformers", "peft", "pandas", "scikit-learn"):
        try:
            packages[name] = package_version(name)
        except Exception:
            packages[name] = "unknown"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "packages": packages,
    }


class TrainingPerformanceCallback(Callback):
    """Record epoch throughput and peak CUDA memory like the source notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.started_at: float | None = None
        self.samples_seen = 0

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        del trainer
        self.samples_seen = 0
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch_idx
        labels = batch.get("gesture_labels")
        if torch.is_tensor(labels):
            self.samples_seen += int(labels.shape[0])

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        del trainer
        if self.started_at is None:
            return
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        seconds = time.perf_counter() - self.started_at
        allocated = reserved = 0.0
        if pl_module.device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(pl_module.device) / (1024**3)
            reserved = torch.cuda.max_memory_reserved(pl_module.device) / (1024**3)
        pl_module.log_dict(
            {
                "train_epoch_seconds": seconds,
                "train_samples_per_second": self.samples_seen / max(seconds, 1e-9),
                "train_peak_allocated_gib": allocated,
                "train_peak_reserved_gib": reserved,
            },
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.started_at = None


def _callbacks(checkpoint_dir: Path, has_validation: bool) -> list[Any]:
    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch-{epoch:03d}",
            # Keep every epoch, not just the last: a run terminated part-way then costs
            # only the epoch in flight, and any epoch can be re-evaluated afterwards.
            save_top_k=-1,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        )
    ]
    callbacks.append(TrainingPerformanceCallback())
    if ENABLE_EARLY_STOPPING:
        if not has_validation:
            raise ValueError("Early stopping requires a validation dataset")
        callbacks.append(
            EarlyStopping(
                monitor="val_joint_loss",
                mode="min",
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
            )
        )
    return callbacks


def resolve_training_stop_status(
    completed_epochs: int, max_epochs: int, early_stopping_triggered: bool
) -> tuple[bool, str]:
    if completed_epochs >= max_epochs:
        return False, "max_epochs"
    if early_stopping_triggered:
        return True, "early_stopping"
    return False, "incomplete"


def _last_logged_metric(history: pd.DataFrame, column: str) -> float | None:
    if column not in history:
        return None
    values = pd.to_numeric(history[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _best_logged_metric(
    history: pd.DataFrame, column: str
) -> tuple[float | None, int | None]:
    if column not in history or "epoch" not in history:
        return None, None
    values = history[["epoch", column]].copy()
    values[column] = pd.to_numeric(values[column], errors="coerce")
    values["epoch"] = pd.to_numeric(values["epoch"], errors="coerce")
    values = values.dropna()
    if values.empty:
        return None, None
    best = values.loc[values[column].idxmin()]
    return float(best[column]), int(best["epoch"]) + 1


def _gpu_smoke_test(
    module: QwenLightningModule,
    batch: Mapping[str, Any],
    configured_batch_size: int,
) -> None:
    """Run a complete configured-batch forward/backward preflight."""
    module.cuda().train()
    device_batch = {
        key: value.to(module.device) if torch.is_tensor(value) else list(value)
        for key, value in batch.items()
    }
    batch_size = int(device_batch["gesture_labels"].shape[0])
    print(
        f"GPU preflight batch: observed={batch_size}, "
        f"configured={configured_batch_size}"
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, predicted_fma = module(**model_batch(device_batch))
            losses = compute_joint_loss(
                logits,
                predicted_fma,
                device_batch["gesture_labels"],
                device_batch["ue_fma_score"],
            )
        losses.total.backward()
    except torch.cuda.OutOfMemoryError as error:
        module.zero_grad(set_to_none=True)
        module.cpu()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"Full-batch GPU preflight failed for batch {batch_size}; training was not started"
        ) from error
    assert logits.shape == (batch_size, TARGET_WINDOW_SIZE, NUM_PRIMITIVE_CLASSES)
    assert predicted_fma.shape == (batch_size,)
    assert torch.all((predicted_fma >= UEFMA_MIN) & (predicted_fma <= UEFMA_MAX))
    gradient_groups = {
        "primitive head": list(module.model.primitive_head.parameters()),
        "regression head": list(module.model.regression_head.parameters()),
        "primitive queries": [module.model.primitive_queries],
        "UE-FMA query": [module.model.fma_query],
        "visual adapter": list(module.model.feature_adapter.parameters()),
        "LoRA parameters": [
            parameter
            for name, parameter in module.model.backbone.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ],
    }
    if module.model.use_patient_prefix:
        assert module.model.patient_encoder is not None
        gradient_groups["patient encoder"] = list(module.model.patient_encoder.parameters())
    missing_gradients = [
        name
        for name, parameters in gradient_groups.items()
        if not parameters or not any(parameter.grad is not None for parameter in parameters)
    ]
    if missing_gradients:
        raise AssertionError(f"Joint loss did not reach trainable groups: {missing_gradients}")
    expected_visual_width = (
        CONDITIONED_VISION_DIM if module.model.use_patient_prefix else QWEN_HIDDEN_SIZE
    )
    assert module.model.last_tensor_shapes["visual_projection"] == (
        batch_size,
        CONTEXT_WINDOW_SIZE,
        expected_visual_width,
    )
    expected_patient_shape = (
        (batch_size, CONTEXT_WINDOW_SIZE, PATIENT_PREFIX_DIM)
        if module.model.use_patient_prefix
        else None
    )
    assert module.model.last_tensor_shapes["patient_prefix"] == expected_patient_shape
    assert module.model.last_tensor_shapes["query_tokens"] == (
        batch_size,
        TARGET_WINDOW_SIZE + 1,
        QWEN_HIDDEN_SIZE,
    )
    module.zero_grad(set_to_none=True)
    module.cpu()
    # logits/predicted_fma still reference the autograd graph; drop them before the
    # cache is released or the preflight's activations stay resident.
    del device_batch, logits, predicted_fma, losses, gradient_groups
    free_gpu_memory("preflight")


def train_final_checkpoint(
    config: RuntimeConfig,
    run_dir: Path,
    run_name: str,
    conditioning: str,
    train_dataset: StrokeRehabDataset,
    validation_dataset: Optional[StrokeRehabDataset],
    collator: StrokeRehabCollator,
    seed: int,
    extra_callbacks: Sequence[Callback] = (),
) -> tuple[Path, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Qwen3.5-0.8B training")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_id = checkpoint_configuration_id(config, conditioning)
    checkpoint_dir = run_dir / "checkpoints" / config_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.ckpt"
    train_manifest_path = run_dir / "training_manifest.json"
    expected_train_subjects = sorted(train_dataset.subjects)
    expected_validation_subjects = (
        sorted(validation_dataset.subjects) if validation_dataset is not None else []
    )
    if train_manifest_path.is_file() and last_checkpoint.is_file():
        manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        same_experiment = (
            manifest.get("training_complete")
            and manifest.get("config_id") == config_id
            and manifest.get("train_subjects") == expected_train_subjects
            and manifest.get("validation_subjects") == expected_validation_subjects
        )
        completed_epochs = int(manifest.get("completed_epochs") or 0)
        if same_experiment and completed_epochs >= config.epochs:
            manifest["reused_completed_checkpoint"] = True
            return last_checkpoint, manifest
        if same_experiment:
            # Same run, fewer epochs than now requested: extend it rather than start
            # over. Lightning restores the optimizer, scheduler and epoch counter from
            # last.ckpt and trains the remainder.
            print(
                f"Extending {run_name} from {completed_epochs} to {config.epochs} epochs "
                f"using {last_checkpoint}"
            )
        elif manifest.get("config_id") == config_id:
            raise RuntimeError("Existing checkpoint subject split is incompatible with this run")

    reset_run_seed(seed)
    train_loader = make_loader(train_dataset, collator, config, shuffle=True, seed=seed)
    val_loader = (
        make_loader(validation_dataset, collator, config, shuffle=False, seed=seed)
        if validation_dataset is not None
        else None
    )
    module = QwenLightningModule(
        run_name=run_name,
        conditioning=conditioning,
        model_name=config.model_name,
        learning_rate=config.learning_rate,
    )
    first_batch = next(iter(train_loader))
    _gpu_smoke_test(module, first_batch, config.batch_size)
    reset_run_seed(seed)
    logger = CSVLogger(save_dir=run_dir / "logs", name=config_id, version="")
    callbacks = [
        *_callbacks(checkpoint_dir, validation_dataset is not None),
        *extra_callbacks,
    ]
    early_stopping_callback = next(
        (callback for callback in callbacks if isinstance(callback, EarlyStopping)),
        None,
    )
    trainer = L.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        precision=training_precision(),
        accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS,
        gradient_clip_val=GRAD_CLIP_NORM,
        deterministic=DETERMINISTIC_TRAINING,
        log_every_n_steps=50,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    resume_path = str(last_checkpoint) if config.resume and last_checkpoint.is_file() else None
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_path,
    )
    if not last_checkpoint.is_file():
        raise RuntimeError(f"Training did not create final checkpoint: {last_checkpoint}")
    history_path = Path(logger.log_dir) / "metrics.csv"
    if not history_path.is_file():
        raise RuntimeError(f"Training metrics were not created: {history_path}")
    history = pd.read_csv(history_path)
    epoch_values = history["epoch"].dropna().astype(int)
    completed_epochs = int(epoch_values.max()) + 1 if not epoch_values.empty else 0
    early_stopping_triggered = bool(
        early_stopping_callback is not None
        and (trainer.should_stop or int(early_stopping_callback.stopped_epoch) > 0)
    )
    early_stopped, stop_reason = resolve_training_stop_status(
        completed_epochs, config.epochs, early_stopping_triggered
    )
    if stop_reason == "incomplete":
        raise RuntimeError(f"Only {completed_epochs}/{config.epochs} epochs were logged")
    best_val_joint_loss, best_val_epoch = _best_logged_metric(history, "val_joint_loss")
    final_metrics = {
        name: _last_logged_metric(history, name)
        for name in (
            "train_joint_loss",
            "train_primitive_loss",
            "train_ue_fma_normalized_mse",
            "train_accuracy",
            "val_joint_loss",
            "val_primitive_loss",
            "val_ue_fma_normalized_mse",
            "val_accuracy",
        )
    }
    manifest = {
        "training_complete": True,
        "config_id": config_id,
        "run_name": run_name,
        "conditioning": conditioning,
        "conditioning_mode": conditioning_mode_label(conditioning),
        "use_patient_prefix": conditioning == CONDITIONING_PREFIX,
        # No metric on the held-out prediction split ever selects a checkpoint or stops
        # training; per-epoch test numbers are reported for reading only.
        "test_used_for_selection": False,
        "epochs": config.epochs,
        "max_epochs": config.epochs,
        "completed_epochs": completed_epochs,
        "learning_rate": config.learning_rate,
        "resumed_from_checkpoint": resume_path is not None,
        "reused_completed_checkpoint": False,
        "early_stopping_enabled": early_stopping_callback is not None,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "early_stopping_monitor": "val_joint_loss",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "early_stopping_best_score": best_val_joint_loss,
        "early_stopping_best_epoch": best_val_epoch,
        "early_stopping_stopped_after_epoch": completed_epochs if early_stopped else None,
        "early_stopping_wait_count": (
            int(early_stopping_callback.wait_count)
            if early_stopping_callback is not None
            else None
        ),
        "final_metrics": final_metrics,
        "checkpoint_selection": "none; final last.ckpt",
        "final_checkpoint": portable_path(last_checkpoint),
        "history_path": str(history_path),
        "seed": seed,
        "train_subjects": sorted(train_dataset.subjects),
        "validation_subjects": (
            sorted(validation_dataset.subjects) if validation_dataset is not None else []
        ),
        "joint_loss": {
            "primitive": "cross_entropy",
            "ue_fma": "MSE(predicted/66,actual/66)",
            "ue_fma_weight": UEFMA_LOSS_WEIGHT,
        },
        "model_metadata": module.model_metadata,
        "runtime": _runtime_environment(),
        "config": config.to_dict(),
        "stage": config.stage,
        "experiment_group": config.experiment_group,
    }
    train_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del module, trainer, train_loader, val_loader, first_batch, callbacks, logger
    free_gpu_memory(f"training {run_name}")
    return last_checkpoint, manifest


def aggregate_uefma_by_subject(window_frame: pd.DataFrame) -> pd.DataFrame:
    actual_ranges = window_frame.groupby("subject_id")["actual_ue_fma"].agg(
        lambda values: float(values.max() - values.min())
    )
    if (actual_ranges > 1e-6).any():
        raise ValueError("UE-FMA changes within at least one subject")
    subject = (
        window_frame.groupby("subject_id", as_index=False)
        .agg(
            actual_ue_fma=("actual_ue_fma", "mean"),
            predicted_ue_fma=("predicted_ue_fma", "mean"),
            run_name=("run_name", "first"),
            fold_id=("fold_id", "first"),
            split=("split", "first"),
        )
    )
    subject["absolute_error"] = (
        subject["actual_ue_fma"] - subject["predicted_ue_fma"]
    ).abs()
    return subject[list(UEFMA_EXPORT_COLUMNS)]


def prediction_is_current(
    existing: Mapping[str, Any],
    config_id: str,
    checkpoint_path: Path,
    run_name: str,
    fold_id: int,
    split: str,
    subjects: Sequence[str],
    completed_epochs: int,
) -> bool:
    """Whether a stored prediction manifest still describes the run being asked for.

    `completed_epochs` is what makes this correct after the epoch count left the
    checkpoint id: `last.ckpt` keeps its path when a run is extended, so every other
    field would still match and the shorter run's predictions would be reused.
    """
    return bool(
        existing.get("prediction_complete")
        and existing.get("config_id") == config_id
        and existing.get("checkpoint") == portable_path(checkpoint_path)
        and existing.get("run_name") == run_name
        and existing.get("fold_id") == fold_id
        and existing.get("split") == split
        and existing.get("subjects") == sorted(subjects)
        and existing.get("completed_epochs") == completed_epochs
    )


def build_prediction_frames(
    prediction_batches: Iterable[Mapping[str, Any]],
    run_name: str,
    fold_id: int,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Turn `predict_step` outputs into the primitive, UE-FMA, and combined frames.

    Shared by `predict_checkpoint` and by the per-epoch metrics callback, which runs the
    same forward pass against the in-memory module instead of a saved checkpoint.
    """
    primitive_records: list[dict[str, Any]] = []
    fma_records: list[dict[str, Any]] = []
    for batch in prediction_batches:
        labels = batch["labels"].numpy()
        predicted = batch["predicted_ids"].numpy()
        probabilities = batch["predicted_probabilities"].numpy()
        actual_fma = batch["actual_fma"].numpy()
        predicted_fma = batch["predicted_fma"].numpy()
        for sample_index in range(labels.shape[0]):
            subject_id = str(batch["subject_id"][sample_index])
            clip_id = str(batch["clip_id"][sample_index])
            task_name = str(batch["task_name"][sample_index])
            target_frame = int(batch["target_frame_index"][sample_index])
            fma_records.append(
                {
                    "subject_id": subject_id,
                    "clip_id": clip_id,
                    "task_name": task_name,
                    "target_frame_index": target_frame,
                    "actual_ue_fma": float(actual_fma[sample_index]),
                    "predicted_ue_fma": float(predicted_fma[sample_index]),
                    "run_name": run_name,
                    "fold_id": fold_id,
                    "split": split,
                }
            )
            for horizon in range(TARGET_WINDOW_SIZE):
                actual_id = int(labels[sample_index, horizon])
                if actual_id == MASK_LABEL_ID:
                    continue
                predicted_id = int(predicted[sample_index, horizon])
                primitive_records.append(
                    {
                        "subject_id": subject_id,
                        "clip_id": clip_id,
                        "task_name": task_name,
                        "target_frame_index": target_frame,
                        "horizon": horizon,
                        "actual_primitive": CLASS_NAMES[actual_id],
                        "predicted_primitive": CLASS_NAMES[predicted_id],
                        "prediction_probability": float(
                            probabilities[sample_index, horizon]
                        ),
                        "run_name": run_name,
                        "fold_id": fold_id,
                        "split": split,
                        "actual_label_id": actual_id,
                        "predicted_label_id": predicted_id,
                        "absolute_frame_index": target_frame + horizon,
                        "prediction_position": POSITION_NAMES[horizon],
                    }
                )
    primitive = pd.DataFrame(primitive_records)
    fma_windows = pd.DataFrame(fma_records)
    if primitive.empty or fma_windows.empty:
        raise RuntimeError("Prediction produced empty output")
    combined = primitive.merge(
        fma_windows[
            [
                "subject_id",
                "clip_id",
                "task_name",
                "target_frame_index",
                "actual_ue_fma",
                "predicted_ue_fma",
            ]
        ],
        on=["subject_id", "clip_id", "task_name", "target_frame_index"],
        how="left",
        validate="many_to_one",
    )
    return primitive, fma_windows, combined


@torch.no_grad()
def predict_checkpoint(
    config: RuntimeConfig,
    checkpoint_path: Path,
    dataset: StrokeRehabDataset,
    collator: StrokeRehabCollator,
    run_name: str,
    conditioning: str,
    fold_id: int,
    split: str,
    output_dir: Path,
    completed_epochs: int,
) -> dict[str, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for prediction")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "oof" if split == "oof" else "test"
    outputs = {
        "primitive": output_dir / f"{prefix}_primitive_predictions.csv",
        "ue_fma": output_dir / f"{prefix}_ue_fma_subject_predictions.csv",
        "combined": output_dir / f"{prefix}_combined_window_predictions.csv",
    }
    complete_path = output_dir / "prediction_manifest.json"
    config_id = checkpoint_configuration_id(config, conditioning)
    if complete_path.is_file() and all(path.is_file() for path in outputs.values()):
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if prediction_is_current(
            existing,
            config_id,
            checkpoint_path,
            run_name,
            fold_id,
            split,
            sorted(dataset.subjects),
            completed_epochs,
        ):
            return outputs

    reset_run_seed(config.seed + max(fold_id, 0))
    loader = make_loader(dataset, collator, config, shuffle=False, seed=config.seed)
    model = QwenLightningModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_conditioning = model.hparams.get("conditioning")
    if checkpoint_conditioning is None:
        raise ValueError(
            f"{checkpoint_path} predates the three-way conditioning axis and cannot be "
            "used; retrain the variant in this output root"
        )
    if str(checkpoint_conditioning) != str(conditioning):
        raise ValueError(
            f"Checkpoint conditioning {checkpoint_conditioning!r} does not match "
            f"requested {conditioning!r}"
        )
    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator="gpu",
        devices=1,
        precision=training_precision(),
        deterministic=DETERMINISTIC_TRAINING,
    )
    prediction_batches = trainer.predict(model, dataloaders=loader)
    primitive, fma_windows, combined = build_prediction_frames(
        prediction_batches, run_name, fold_id, split
    )
    if set(primitive["horizon"]) != set(range(TARGET_WINDOW_SIZE)):
        raise AssertionError("Prediction export is missing configured horizons")
    if set(primitive["prediction_position"]) != set(POSITION_NAMES):
        raise AssertionError("Prediction export has unexpected position labels")
    timeline_keys = ["subject_id", "task_name", "clip_id", "absolute_frame_index"]
    if primitive.duplicated(timeline_keys).any():
        raise AssertionError("Reconstructed predictions contain duplicate frame keys")
    expected_first_target = (CONTEXT_WINDOW_SIZE - 1) * FRAME_SKIP + TARGET_FRAME_OFFSET
    for keys, clip in primitive.groupby(["subject_id", "task_name", "clip_id"], sort=False):
        frames = np.sort(clip["absolute_frame_index"].to_numpy(dtype=int))
        if frames[0] != expected_first_target:
            raise AssertionError(f"Unexpected first target frame for clip {keys}")
        if len(frames) > 1 and not np.all(np.diff(frames) == 1):
            raise AssertionError(f"Reconstructed timeline is not contiguous for clip {keys}")
        for horizon, horizon_frame in clip.groupby("horizon", sort=False):
            horizon_frames = np.sort(
                horizon_frame["absolute_frame_index"].to_numpy(dtype=int)
            )
            if len(horizon_frames) > 1 and not np.all(
                np.diff(horizon_frames) == WINDOW_STRIDE
            ):
                raise AssertionError(
                    f"Position {horizon} does not follow window stride for clip {keys}"
                )
    primitive[list(PRIMITIVE_EXPORT_COLUMNS)].to_csv(outputs["primitive"], index=False)
    subject_fma = aggregate_uefma_by_subject(fma_windows)
    subject_fma.to_csv(outputs["ue_fma"], index=False)
    combined.to_csv(outputs["combined"], index=False)
    complete_path.write_text(
        json.dumps(
            {
                "prediction_complete": True,
                "config_id": config_id,
                "conditioning": conditioning,
                "completed_epochs": completed_epochs,
                "run_name": run_name,
                "fold_id": fold_id,
                "split": split,
                "subjects": sorted(dataset.subjects),
                "primitive_rows": len(primitive),
                "ue_fma_subject_rows": len(subject_fma),
                "checkpoint": portable_path(checkpoint_path),
                "outputs": {key: portable_path(path) for key, path in outputs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del model, trainer, loader, prediction_batches
    free_gpu_memory(f"prediction {run_name}")
    return outputs


def run_experiment(
    config: RuntimeConfig,
    tokenizer: Any,
    run_dir: Path,
    run_name: str,
    conditioning: str,
    train_npz: Path,
    train_subjects: Iterable[str],
    validation_subjects: Iterable[str],
    prediction_npz: Path,
    prediction_subjects: Iterable[str],
    fold_id: int,
    split: str,
    epoch_metrics_callback_factory: Any | None = None,
) -> dict[str, Any]:
    """Train one variant and predict its held-out split.

    `epoch_metrics_callback_factory`, when given, is called with the prediction dataset
    and collator and must return Lightning callbacks; the multi-seed driver uses it to
    write per-epoch metrics. The cross-validation path passes nothing.
    """
    train_subjects = tuple(sorted(train_subjects))
    validation_subjects = tuple(sorted(validation_subjects))
    prediction_subjects = tuple(sorted(prediction_subjects))
    seed = config.seed + max(fold_id, 0)
    normalizer = (
        PatientNormalizer.fit(train_npz, train_subjects, config.cache_root)
        if conditioning == CONDITIONING_PREFIX
        else None
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if normalizer is not None:
        (run_dir / "patient_normalizer.json").write_text(
            json.dumps(normalizer.to_dict(), indent=2), encoding="utf-8"
        )
    train_dataset = StrokeRehabDataset(
        train_npz,
        config.cache_root,
        include_subjects=train_subjects,
        conditioning=conditioning,
        patient_normalizer=normalizer,
    )
    validation_dataset = StrokeRehabDataset(
        train_npz,
        config.cache_root,
        include_subjects=validation_subjects,
        conditioning=conditioning,
        patient_normalizer=normalizer,
    )
    prediction_dataset = StrokeRehabDataset(
        prediction_npz,
        config.cache_root,
        include_subjects=prediction_subjects,
        conditioning=conditioning,
        patient_normalizer=normalizer,
    )
    if train_dataset.subjects & validation_dataset.subjects:
        raise AssertionError("Training and validation subjects overlap")
    if train_dataset.subjects & prediction_dataset.subjects:
        raise AssertionError("Training and prediction subjects overlap")
    if split == "oof" and validation_dataset.subjects != prediction_dataset.subjects:
        raise AssertionError("OOF validation and prediction subjects must match")
    if split == "test" and validation_dataset.subjects & prediction_dataset.subjects:
        raise AssertionError("Full-run validation and test subjects overlap")
    collator = StrokeRehabCollator(tokenizer, MAX_TEXT_LENGTH)
    extra_callbacks = (
        epoch_metrics_callback_factory(prediction_dataset, collator, seed)
        if epoch_metrics_callback_factory is not None
        else ()
    )
    checkpoint, train_manifest = train_final_checkpoint(
        config,
        run_dir,
        run_name,
        conditioning,
        train_dataset,
        validation_dataset,
        collator,
        seed,
        extra_callbacks=extra_callbacks,
    )
    train_manifest = dict(train_manifest)
    train_manifest.update(
        {
            "fold_id": fold_id,
            "split": split,
            "prediction_subjects": sorted(prediction_dataset.subjects),
        }
    )
    (run_dir / "training_manifest.json").write_text(
        json.dumps(train_manifest, indent=2), encoding="utf-8"
    )
    prediction_paths = predict_checkpoint(
        config,
        checkpoint,
        prediction_dataset,
        collator,
        run_name,
        conditioning,
        fold_id,
        split,
        run_dir / "predictions",
        completed_epochs=int(train_manifest["completed_epochs"]),
    )
    return {
        "run_dir": portable_path(run_dir),
        "run_name": run_name,
        "conditioning": conditioning,
        "seed": seed,
        "fold_id": fold_id,
        "split": split,
        "checkpoint": portable_path(checkpoint),
        "training_manifest": train_manifest,
        "predictions": {key: portable_path(value) for key, value in prediction_paths.items()},
    }
