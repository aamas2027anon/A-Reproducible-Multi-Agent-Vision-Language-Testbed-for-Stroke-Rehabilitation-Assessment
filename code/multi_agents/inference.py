"""The trained movement model, loaded outside Lightning, and its on-disk cache.

Only the perception agent holds a `MovementModelRunner`. The network comes from
`train/model.build_model` (Qwen3.5-0.8B-Base + LoRA + adapters + heads), so it is
the trained architecture itself. The Lightning checkpoint stores only trainable
tensors ("compact" state), prefixed `model.`; the frozen Base weights are loaded from
`--base-model`.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from multi_agents.settings import CHECKPOINT_DIR, DEFAULT_CHECKPOINT, PRIMITIVES
from multi_agents.state import PerceptionReport
from multi_agents.train_bridge import PatientNormalizer, StrokeRehabCollator, build_model, train_config


@dataclass(frozen=True)
class WindowResult:
    """Model output for one window: frames t..t+4 and a UE-FMA score."""

    probs: np.ndarray  # [5 horizons, 5 classes]
    ue_fma: float
    target_frame: int  # t, relative to clip start
    ok: bool  # False = non-finite input, model not run
    model_ms: float  # model compute attributed to this window


def window_report(window_index: int, result: WindowResult) -> PerceptionReport:
    return PerceptionReport(
        window_index=window_index,
        target_frame_index=int(result.target_frame),
        probs=[{name: float(p) for name, p in zip(PRIMITIVES, row)} for row in result.probs],
        predicted_labels=[PRIMITIVES[i] for i in result.probs.argmax(axis=1)],
        confidence=[float(v) for v in result.probs.max(axis=1)],
        ue_fma=float(result.ue_fma),
        integrity_ok=bool(result.ok),
    )


class WindowAssembler:
    """Turns a live frame stream into the training pipeline's windows.

    `StrokeRehabDataset` places windows at starts 0, 5, 10, ... of a clip, each
    covering 116 frames and sampling 24 of them (every 5th). Emitting a window when
    116 frames are buffered and (frames seen - 116) is a multiple of the stride yields
    exactly those windows, so live predictions equal the offline ones.
    """

    def __init__(self, text: str, patient_features: torch.Tensor | None = None) -> None:
        if train_config.TARGET_FRAME_OFFSET != 0:
            raise NotImplementedError("Streaming assumes the target is the last context frame")
        self.offsets = np.arange(train_config.CONTEXT_WINDOW_SIZE) * train_config.FRAME_SKIP
        self.span = int(self.offsets[-1]) + 1
        self.stride = train_config.WINDOW_STRIDE
        self.text = text
        self.patient_features = patient_features
        self.buffer: deque[np.ndarray] = deque(maxlen=self.span)
        self.frames_seen = 0
        self.pending: tuple[dict, int] | None = None

    def push(self, frame: np.ndarray) -> bool:
        """Add one frame; return True when a new window is ready."""
        self.buffer.append(np.asarray(frame, dtype=np.float32))
        self.frames_seen += 1
        start = self.frames_seen - self.span
        if start < 0 or start % self.stride:
            return False
        context = np.stack(self.buffer)[self.offsets]
        sample = {"text": self.text, "vision_feat": torch.from_numpy(np.ascontiguousarray(context))}
        if self.patient_features is not None:
            sample["patient_features"] = self.patient_features
        self.pending = (sample, self.frames_seen - 1)
        return True

    def take(self) -> tuple[dict, int]:
        if self.pending is None:
            raise RuntimeError("No complete window is waiting")
        pending, self.pending = self.pending, None
        return pending


@dataclass
class ExerciseInference:
    """Model outputs for every window of one exercise."""

    probs: np.ndarray  # [W, 5 horizons, 5 classes]
    ue_fma: np.ndarray  # [W]
    target_frames: np.ndarray  # [W], relative to clip start
    window_ok: np.ndarray  # [W] bool; False = non-finite input, model not run
    latency_ms: float

    @property
    def n_windows(self) -> int:
        return int(len(self.ue_fma))

    def window(self, window_index: int) -> WindowResult:
        return WindowResult(
            probs=self.probs[window_index],
            ue_fma=float(self.ue_fma[window_index]),
            target_frame=int(self.target_frames[window_index]),
            ok=bool(self.window_ok[window_index]),
            model_ms=self.latency_ms / max(self.n_windows, 1),
        )

    def report(self, window_index: int) -> PerceptionReport:
        return window_report(window_index, self.window(window_index))


def find_checkpoint(path: str | Path | None) -> Path:
    """--checkpoint, else `checkpoint` in project_config.toml, else the only .ckpt there."""
    if path is None:
        path = DEFAULT_CHECKPOINT
    if path is not None:
        checkpoint = Path(path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint.resolve()
    candidates = sorted(CHECKPOINT_DIR.glob("*.ckpt"))
    if len(candidates) != 1:
        found = ", ".join(c.name for c in candidates) or "none"
        raise FileNotFoundError(
            f"Expected exactly one .ckpt in {CHECKPOINT_DIR} (found: {found}); pass --checkpoint"
        )
    return candidates[0].resolve()


def checkpoint_identity(checkpoint: Path, base_model_dir: Path) -> str:
    stat = Path(checkpoint).stat()
    payload = f"{Path(checkpoint).resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{Path(base_model_dir).name}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_checkpoint_conditioning(checkpoint: Path) -> tuple[str, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = dict(payload.get("hyper_parameters") or {})
    conditioning = hparams.get("conditioning")
    if conditioning not in train_config.CONDITIONING_MODES:
        raise ValueError(
            f"{checkpoint} has no valid hyper_parameters.conditioning ({conditioning!r}); "
            "expected a checkpoint from train/training.py"
        )
    return str(conditioning), hparams


def load_patient_normalizer(checkpoint: Path, conditioning: str) -> PatientNormalizer | None:
    if conditioning != train_config.CONDITIONING_PREFIX:
        return None
    # Next to the checkpoint (a copied checkpoint), else in the training run directory:
    # <run>/checkpoints/<config id>/<name>.ckpt -> <run>/patient_normalizer.json.
    checkpoint = Path(checkpoint).resolve()
    candidates = [checkpoint.with_name("patient_normalizer.json")]
    if len(checkpoint.parents) > 2:
        candidates.append(checkpoint.parents[2] / "patient_normalizer.json")
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"The prefix-conditioned checkpoint needs patient_normalizer.json (looked in "
            f"{', '.join(str(c.parent) for c in candidates)}); copy it from the training run directory"
        )
    values = json.loads(path.read_text(encoding="utf-8"))
    values["fitted_subjects"] = tuple(values["fitted_subjects"])
    return PatientNormalizer(**values)


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def window_integrity(samples: list[dict]) -> np.ndarray:
    ok = []
    for sample in samples:
        finite = bool(torch.isfinite(sample["vision_feat"]).all())
        if "patient_features" in sample:
            finite = finite and bool(torch.isfinite(sample["patient_features"]).all())
        ok.append(finite)
    return np.asarray(ok, dtype=bool)


# Fields `StrokeRehabCollator` requires besides the model inputs (text, vision_feat,
# patient_features). None of them reaches the model (`model.model_batch` whitelists inputs).
_COLLATOR_PLACEHOLDERS = {"subject_id": "", "clip_id": "", "task_name": "", "target_frame_index": -1}
_NO_TARGETS = {
    "gesture_labels": torch.zeros(train_config.TARGET_WINDOW_SIZE, dtype=torch.long),
    "ue_fma_score": torch.tensor(0.0),
}


class MovementModelRunner:
    def __init__(
        self,
        checkpoint: Path,
        base_model_dir: Path,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        from transformers import AutoTokenizer

        self.checkpoint = Path(checkpoint)
        self.conditioning, self.hparams = read_checkpoint_conditioning(self.checkpoint)
        self.normalizer = load_patient_normalizer(self.checkpoint, self.conditioning)
        self.batch_size = int(batch_size)
        self.device = select_device(device)

        model, self.metadata = build_model(
            str(base_model_dir), self.conditioning == train_config.CONDITIONING_PREFIX
        )
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not payload.get("compact_trainable_state"):
            warnings.warn(f"{self.checkpoint} is not a compact training checkpoint", stacklevel=2)
        state_dict = {
            key.removeprefix("model."): value
            for key, value in payload["state_dict"].items()
            if key.startswith("model.")
        }
        result = model.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Checkpoint has keys the model lacks: {result.unexpected_keys[:10]}")
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing = sorted(trainable - state_dict.keys())
        if missing:
            raise ValueError(f"Checkpoint is missing trained tensors: {missing[:10]}")
        del payload, state_dict
        self.model = model.to(self.device).eval()

        # Mirrors training.build_tokenizer, which is not imported to avoid Lightning.
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.collator = StrokeRehabCollator(tokenizer, train_config.MAX_TEXT_LENGTH)

    @torch.inference_mode()
    def predict(
        self, samples: list[dict], target_frames: np.ndarray, progress: str | None = None
    ) -> ExerciseInference:
        """Score windows in batches; `progress` names a progress bar (None = silent)."""
        started = time.perf_counter()
        count = len(samples)
        probs = np.full((count, train_config.TARGET_WINDOW_SIZE, len(PRIMITIVES)), 1.0 / len(PRIMITIVES), np.float32)
        ue_fma = np.full(count, np.nan, dtype=np.float32)
        window_ok = window_integrity(samples)
        valid = np.flatnonzero(window_ok)
        bar = tqdm(total=len(valid), desc=progress, unit="window", leave=False, dynamic_ncols=True) if progress else None
        for offset in range(0, len(valid), self.batch_size):
            chunk = valid[offset : offset + self.batch_size]
            if bar is not None:
                bar.update(len(chunk))
            # The training collator also stacks targets and copies bookkeeping fields the
            # model never reads. Placeholders fill them, so ground truth stays out of here
            # and live (streamed) windows, which carry only the model inputs, collate too.
            batch = self.collator([{**_COLLATOR_PLACEHOLDERS, **samples[i]} | _NO_TARGETS for i in chunk])
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "text_attention_mask": batch["text_attention_mask"].to(self.device),
                "vision_feat": batch["vision_feat"].to(self.device),
                "patient_features": (
                    batch["patient_features"].to(self.device) if "patient_features" in batch else None
                ),
            }
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                logits, predicted_fma = self.model(**inputs)
            probs[chunk] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            ue_fma[chunk] = predicted_fma.float().cpu().numpy()
        if bar is not None:
            bar.close()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return ExerciseInference(
            probs=probs,
            ue_fma=ue_fma,
            target_frames=np.asarray(target_frames, dtype=np.int64),
            window_ok=window_ok,
            latency_ms=1000.0 * (time.perf_counter() - started),
        )


class InferenceCache:
    """One `.npz` per exercise, valid only for the checkpoint that produced it."""

    def __init__(self, root: Path, identity: str) -> None:
        self.root = Path(root) / identity
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key.replace('/', '__')}.npz"

    def get(self, key: str) -> ExerciseInference | None:
        path = self._path(key)
        if not path.is_file():
            return None
        with np.load(path) as data:
            return ExerciseInference(
                probs=data["probs"],
                ue_fma=data["ue_fma"],
                target_frames=data["target_frames"],
                window_ok=data["window_ok"],
                latency_ms=float(data["latency_ms"]),
            )

    def put(self, key: str, result: ExerciseInference) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)  # survives the output folder being cleared mid-run
        temporary = path.with_name(f".{path.stem}.partial.npz")
        np.savez(
            temporary,
            probs=result.probs,
            ue_fma=result.ue_fma,
            target_frames=result.target_frames,
            window_ok=result.window_ok,
            latency_ms=np.float64(result.latency_ms),
        )
        temporary.replace(path)
