from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CONDITIONING_MODE,
    CONDITIONED_VISION_DIM,
    CONTEXT_WINDOW_SIZE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MASK_LABEL_ID,
    NUM_PRIMITIVE_CLASSES,
    PATIENT_FEATURE_DIM,
    PATIENT_PREFIX_DIM,
    QWEN_HIDDEN_SIZE,
    TARGET_WINDOW_SIZE,
    UEFMA_LOSS_WEIGHT,
    UEFMA_MAX,
    VISION_FEATURE_DIM,
    USE_GRADIENT_CHECKPOINTING,
)


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    primitive: torch.Tensor
    ue_fma_normalized: torch.Tensor


def compute_joint_loss(
    primitive_logits: torch.Tensor,
    predicted_fma: torch.Tensor,
    primitive_targets: torch.Tensor,
    actual_fma: torch.Tensor,
    ue_fma_weight: float = UEFMA_LOSS_WEIGHT,
) -> LossBundle:
    if primitive_logits.shape[:2] != primitive_targets.shape:
        raise ValueError(
            f"Logit/target shape mismatch: {primitive_logits.shape} vs {primitive_targets.shape}"
        )
    if predicted_fma.shape != actual_fma.shape:
        raise ValueError(f"UE-FMA shape mismatch: {predicted_fma.shape} vs {actual_fma.shape}")
    primitive = F.cross_entropy(
        primitive_logits.reshape(-1, NUM_PRIMITIVE_CLASSES),
        primitive_targets.reshape(-1),
        ignore_index=MASK_LABEL_ID,
    )
    ue_fma_normalized = F.mse_loss(
        predicted_fma.float() / UEFMA_MAX,
        actual_fma.float() / UEFMA_MAX,
    )
    total = primitive + float(ue_fma_weight) * ue_fma_normalized
    if not all(torch.isfinite(value) for value in (total, primitive, ue_fma_normalized)):
        raise FloatingPointError("Non-finite joint training loss")
    return LossBundle(total=total, primitive=primitive, ue_fma_normalized=ue_fma_normalized)


def model_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
    """Whitelist inputs; target and identifying metadata cannot enter the model."""
    return {
        "input_ids": batch["input_ids"],
        "text_attention_mask": batch["text_attention_mask"],
        "vision_feat": batch["vision_feat"],
        "patient_features": batch.get("patient_features"),
    }


class QwenMovementModel(nn.Module):
    """Qwen model with per-frame patient prefixes and parallel supervised heads."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        use_patient_prefix: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.use_patient_prefix = bool(use_patient_prefix)
        if self.hidden_size != QWEN_HIDDEN_SIZE:
            raise ValueError(
                f"This pipeline requires Qwen hidden size {QWEN_HIDDEN_SIZE}, got {hidden_size}"
            )
        vision_output_dim = CONDITIONED_VISION_DIM if self.use_patient_prefix else hidden_size
        self.feature_adapter = nn.Sequential(
            nn.LayerNorm(VISION_FEATURE_DIM),
            nn.Linear(VISION_FEATURE_DIM, vision_output_dim),
            nn.GELU(),
            nn.LayerNorm(vision_output_dim),
        )
        self.patient_encoder: nn.Module | None = None
        if self.use_patient_prefix:
            self.patient_encoder = nn.Sequential(
                nn.LayerNorm(PATIENT_FEATURE_DIM),
                nn.Linear(PATIENT_FEATURE_DIM, PATIENT_PREFIX_DIM),
                nn.GELU(),
                nn.LayerNorm(PATIENT_PREFIX_DIM),
            )
        self.temporal_embedding = nn.Parameter(
            torch.empty(CONTEXT_WINDOW_SIZE, hidden_size)
        )
        self.primitive_queries = nn.Parameter(
            torch.empty(TARGET_WINDOW_SIZE, hidden_size)
        )
        self.fma_query = nn.Parameter(torch.empty(1, hidden_size))
        self.primitive_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, NUM_PRIMITIVE_CLASSES),
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.last_tensor_shapes: dict[str, Any] = {}
        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.primitive_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.fma_query, mean=0.0, std=0.02)

    def _visual_tokens(
        self,
        vision_feat: torch.Tensor,
        patient_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        visual_projection = self.feature_adapter(vision_feat)
        patient_prefix: torch.Tensor | None = None
        if self.use_patient_prefix:
            if patient_features is None:
                raise ValueError("Patient-prefix model requires patient_features")
            if patient_features.shape != (vision_feat.shape[0], PATIENT_FEATURE_DIM):
                raise ValueError(f"Unexpected patient shape: {tuple(patient_features.shape)}")
            assert self.patient_encoder is not None
            encoded_patient = self.patient_encoder(patient_features)
            patient_prefix = encoded_patient.unsqueeze(1).expand(
                -1, CONTEXT_WINDOW_SIZE, -1
            )
            visual_tokens = torch.cat((patient_prefix, visual_projection), dim=-1)
        elif patient_features is not None:
            raise ValueError("Base model must not receive patient_features")
        else:
            visual_tokens = visual_projection
        if visual_tokens.shape[-1] != self.hidden_size:
            raise AssertionError(
                f"Fused visual width is {visual_tokens.shape[-1]}, expected {self.hidden_size}"
            )
        return visual_tokens, visual_projection, patient_prefix

    def forward(
        self,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        vision_feat: torch.Tensor,
        patient_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vision_feat.ndim != 3 or vision_feat.shape[1:] != (
            CONTEXT_WINDOW_SIZE,
            VISION_FEATURE_DIM,
        ):
            raise ValueError(f"Unexpected vision shape: {tuple(vision_feat.shape)}")
        if input_ids.shape != text_attention_mask.shape:
            raise ValueError("input_ids and text_attention_mask must have the same shape")

        embedding_layer = self.backbone.get_input_embeddings()
        text_tokens = embedding_layer(input_ids)
        visual_tokens, visual_projection, patient_prefix = self._visual_tokens(
            vision_feat, patient_features
        )
        visual_tokens = (visual_tokens + self.temporal_embedding.unsqueeze(0)).to(
            text_tokens.dtype
        )
        batch_size = input_ids.shape[0]
        primitive_queries = self.primitive_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        fma_query = self.fma_query.unsqueeze(0).expand(batch_size, -1, -1)
        query_tokens = torch.cat((primitive_queries, fma_query), dim=1).to(
            text_tokens.dtype
        )
        inputs_embeds = torch.cat((text_tokens, visual_tokens, query_tokens), dim=1)
        suffix_mask = torch.ones(
            batch_size,
            CONTEXT_WINDOW_SIZE + TARGET_WINDOW_SIZE + 1,
            dtype=text_attention_mask.dtype,
            device=text_attention_mask.device,
        )
        attention_mask = torch.cat((text_attention_mask, suffix_mask), dim=1)
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        query_states = outputs.last_hidden_state[:, -(TARGET_WINDOW_SIZE + 1):]
        primitive_states = query_states[:, :TARGET_WINDOW_SIZE]
        fma_state = query_states[:, -1]
        primitive_logits = self.primitive_head(primitive_states).float()
        raw_fma = self.regression_head(fma_state).squeeze(-1).float()
        predicted_fma = UEFMA_MAX * torch.sigmoid(raw_fma)
        self.last_tensor_shapes = {
            "input_ids": tuple(input_ids.shape),
            "visual_features": tuple(vision_feat.shape),
            "visual_projection": tuple(visual_projection.shape),
            "patient_prefix": (
                tuple(patient_prefix.shape) if patient_prefix is not None else None
            ),
            "fused_visual_tokens": tuple(visual_tokens.shape),
            "query_tokens": tuple(query_tokens.shape),
            "fused_inputs": tuple(inputs_embeds.shape),
            "primitive_states": tuple(primitive_states.shape),
            "fma_state": tuple(fma_state.shape),
            "primitive_logits": tuple(primitive_logits.shape),
            "ue_fma_predictions": tuple(predicted_fma.shape),
        }
        return primitive_logits, predicted_fma


def preferred_model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def preferred_attn_implementation(dtype: torch.dtype | None = None) -> str:
    """FlashAttention-2 for Qwen's full-attention layers when installed, else SDPA.

    The Gated DeltaNet (linear-attention) layers pick up the `fla` and `causal_conv1d`
    kernels automatically when those packages import; nothing to request here.
    """
    from transformers.utils import is_flash_attn_2_available

    dtype = preferred_model_dtype() if dtype is None else dtype
    if dtype in (torch.bfloat16, torch.float16) and is_flash_attn_2_available():
        return "flash_attention_2"
    return "sdpa"


def load_qwen_language_backbone(model_name: str) -> nn.Module:
    from transformers import AutoModelForMultimodalLM

    dtype = preferred_model_dtype()
    pretrained = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=preferred_attn_implementation(dtype),
    )
    if not hasattr(pretrained, "model") or not hasattr(pretrained.model, "language_model"):
        raise AttributeError(
            f"{model_name} did not expose model.language_model; verify Transformers compatibility"
        )
    language_model = pretrained.model.language_model
    del pretrained
    gc.collect()
    for parameter in language_model.parameters():
        parameter.requires_grad = False
    language_model.config.use_cache = False
    if USE_GRADIENT_CHECKPOINTING:
        language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif hasattr(language_model, "gradient_checkpointing_disable"):
        language_model.gradient_checkpointing_disable()
    return language_model


def validate_lora_targets(backbone: nn.Module) -> dict[str, int]:
    counts = {
        suffix: sum(
            isinstance(module, nn.Linear) and name.endswith(suffix)
            for name, module in backbone.named_modules()
        )
        for suffix in LORA_TARGET_MODULES
    }
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Configured LoRA modules not found in Qwen: {missing}")
    return counts


def parameter_report(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "trainable": int(trainable),
        "total": int(total),
        "trainable_percent": 100.0 * trainable / total,
    }


def build_model(
    model_name: str,
    use_patient_prefix: bool,
) -> tuple[QwenMovementModel, dict[str, Any]]:
    from peft import LoraConfig, TaskType, get_peft_model

    backbone = load_qwen_language_backbone(model_name)
    target_counts = validate_lora_targets(backbone)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(LORA_TARGET_MODULES),
    )
    backbone = get_peft_model(backbone, lora_config)
    hidden_size = int(backbone.get_base_model().config.hidden_size)
    model = QwenMovementModel(backbone, hidden_size, use_patient_prefix)
    invalid = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if invalid:
        raise AssertionError(f"Non-LoRA backbone parameters are trainable: {invalid[:10]}")
    return model, {
        "model_name": model_name,
        "hidden_size": hidden_size,
        "conditioning": (
            CONDITIONING_MODE
            if use_patient_prefix
            else "none_full_visual_1024"
        ),
        "visual_projection_width": (
            CONDITIONED_VISION_DIM if use_patient_prefix else QWEN_HIDDEN_SIZE
        ),
        "patient_prefix_width": PATIENT_PREFIX_DIM if use_patient_prefix else 0,
        "output_architecture": "parallel_primitive_and_uefma_queries",
        "sequence_layout": (
            "text -> 24 fused visual tokens -> 5 primitive queries -> 1 UE-FMA query"
        ),
        "patient_inputs": (
            ["paretic_side", "gender", "race", "age", "time_since_stroke"]
            if use_patient_prefix
            else []
        ),
        "ue_fma_path": "dedicated_fma_query_1024_to_score",
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": list(LORA_TARGET_MODULES),
            "matched_module_counts": target_counts,
        },
        "parameters": parameter_report(model),
    }
