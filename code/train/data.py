from __future__ import annotations

import os
import random
import shutil
import zipfile
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    CONDITIONING_MODES,
    CONDITIONING_NONE,
    CONDITIONING_PREFIX,
    CONDITIONING_TEXT,
    CONTEXT_WINDOW_SIZE,
    EXPECTED_TEST_SUBJECTS,
    EXPECTED_TRAIN_SUBJECTS,
    FRAME_SKIP,
    GENDER_CATEGORIES,
    LABEL_VOCAB,
    MASK_LABEL_ID,
    PARETIC_SIDE_CATEGORIES,
    PATIENT_FEATURE_DIM,
    PREDICTION_PROMPT,
    RACE_CATEGORIES,
    RuntimeConfig,
    SOURCE_GESTURE_SEQUENCE_LENGTH,
    TARGET_FRAME_OFFSET,
    TARGET_WINDOW_SIZE,
    TASK_DESCRIPTIONS,
    UEFMA_MAX,
    UEFMA_MIN,
    VISION_FEATURE_DIM,
    WINDOW_STRIDE,
)


SHARED_NPZ_KEYS = (
    "vision_feat",
    "gesture_seq",
    "subject_id",
    "UE_FMA_score",
    "task_name",
    "clip_id",
    "frame_index",
)
PATIENT_NPZ_KEYS = (
    "paretic_side",
    "gender",
    "age",
    "race",
    "time_since_stroke",
)


def canonical_subject_id(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("s") and text[1:].isdigit():
        return f"s{int(text[1:]):02d}"
    raise ValueError(f"Invalid subject ID: {value!r}")


def extract_npz_members(
    npz_path: Path,
    cache_root: Path,
    keys: Sequence[str],
) -> dict[str, Path]:
    """Extract selected NPY members once so NumPy can memory-map them."""
    npz_path, cache_root = Path(npz_path), Path(cache_root)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    cache_dir = cache_root / npz_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Path] = {}
    with zipfile.ZipFile(npz_path) as archive:
        names = set(archive.namelist())
        for key in keys:
            member = f"{key}.npy"
            if member not in names:
                raise KeyError(f"{npz_path.name} is missing required member {member}")
            destination = cache_dir / member
            expected_size = archive.getinfo(member).file_size
            if not destination.exists() or destination.stat().st_size != expected_size:
                temporary = destination.with_name(
                    f".{destination.name}.{os.getpid()}.partial"
                )
                temporary.unlink(missing_ok=True)
                try:
                    with archive.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            output[key] = destination
    return output


def npz_subjects(npz_path: Path, cache_root: Path) -> tuple[str, ...]:
    path = extract_npz_members(npz_path, cache_root, ("subject_id",))["subject_id"]
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return tuple(sorted({canonical_subject_id(value) for value in np.unique(values)}))


@dataclass(frozen=True)
class PatientNormalizer:
    age_mean: float
    age_std: float
    time_mean: float
    time_std: float
    fitted_subjects: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        npz_path: Path,
        subjects: Iterable[str],
        cache_root: Path,
    ) -> "PatientNormalizer":
        selected = frozenset(canonical_subject_id(value) for value in subjects)
        keys = ("subject_id", "frame_index", "age", "time_since_stroke")
        paths = extract_npz_members(npz_path, cache_root, keys)
        arrays = {
            key: np.load(path, mmap_mode="r", allow_pickle=False)
            for key, path in paths.items()
        }
        clip_starts = np.flatnonzero(np.asarray(arrays["frame_index"]) == 0)
        records: dict[str, tuple[float, float]] = {}
        for index in clip_starts:
            subject = canonical_subject_id(arrays["subject_id"][index])
            if subject not in selected:
                continue
            values = (
                float(arrays["age"][index]),
                float(arrays["time_since_stroke"][index]),
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite patient metadata for {subject}")
            if subject in records and not np.allclose(records[subject], values):
                raise ValueError(f"Patient metadata changes across clips for {subject}")
            records[subject] = values
        missing = selected - records.keys()
        if missing:
            raise ValueError(f"Cannot fit patient normalizer; missing subjects: {sorted(missing)}")
        ordered = np.asarray([records[subject] for subject in sorted(selected)], dtype=float)
        means = ordered.mean(axis=0)
        stds = ordered.std(axis=0, ddof=0)
        if np.any(np.isclose(stds, 0)):
            raise ValueError("Patient continuous feature has zero training-subject variance")
        return cls(
            age_mean=float(means[0]),
            age_std=float(stds[0]),
            time_mean=float(means[1]),
            time_std=float(stds[1]),
            fitted_subjects=tuple(sorted(selected)),
        )

    def encode(
        self,
        paretic_side: Any,
        gender: Any,
        age: Any,
        race: Any,
        time_since_stroke: Any,
    ) -> np.ndarray:
        pieces = [
            _one_hot(str(paretic_side), PARETIC_SIDE_CATEGORIES, "paretic_side"),
            _one_hot(str(gender), GENDER_CATEGORIES, "gender"),
            _one_hot(str(race), RACE_CATEGORIES, "race"),
            np.asarray([(float(age) - self.age_mean) / self.age_std], dtype=np.float32),
            np.asarray(
                [(float(time_since_stroke) - self.time_mean) / self.time_std],
                dtype=np.float32,
            ),
        ]
        encoded = np.concatenate(pieces).astype(np.float32, copy=False)
        if encoded.shape != (PATIENT_FEATURE_DIM,) or not np.isfinite(encoded).all():
            raise ValueError(f"Invalid patient feature vector: {encoded}")
        return encoded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _one_hot(value: str, categories: Sequence[str], field: str) -> np.ndarray:
    if value not in categories:
        raise ValueError(f"Unknown {field} category: {value!r}")
    encoded = np.zeros(len(categories), dtype=np.float32)
    encoded[categories.index(value)] = 1.0
    return encoded


class StrokeRehabDataset(Dataset):
    """Memory-mapped temporal windows with safe, optional patient features."""

    def __init__(
        self,
        npz_path: Path,
        cache_root: Path,
        include_subjects: Optional[Iterable[str]] = None,
        conditioning: str = CONDITIONING_NONE,
        patient_normalizer: Optional[PatientNormalizer] = None,
    ) -> None:
        super().__init__()
        self.npz_path = Path(npz_path)
        self.include_subjects = (
            None
            if include_subjects is None
            else frozenset(canonical_subject_id(value) for value in include_subjects)
        )
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"Unknown conditioning {conditioning!r}; expected one of {CONDITIONING_MODES}"
            )
        self.conditioning = conditioning
        self.use_patient_prefix = conditioning == CONDITIONING_PREFIX
        self.use_patient_text = conditioning == CONDITIONING_TEXT
        self.patient_normalizer = patient_normalizer
        if self.use_patient_prefix and self.patient_normalizer is None:
            raise ValueError("Prefix-conditioned dataset requires a fitted PatientNormalizer")
        if not self.use_patient_prefix and self.patient_normalizer is not None:
            raise ValueError(
                f"{conditioning!r} conditioning does not normalize patient features"
            )
        # The text variant reads the same NPZ patient members, but as raw values for
        # the prompt rather than as a normalized feature vector.
        needs_patient = conditioning != CONDITIONING_NONE
        keys = SHARED_NPZ_KEYS + (PATIENT_NPZ_KEYS if needs_patient else ())
        paths = extract_npz_members(self.npz_path, cache_root, keys)
        for key, path in paths.items():
            setattr(self, key, np.load(path, mmap_mode="r", allow_pickle=False))
        self.ue_fma_score = self.UE_FMA_score
        self.context_offsets = np.arange(CONTEXT_WINDOW_SIZE, dtype=np.int64) * FRAME_SKIP
        self.required_span = int(self.context_offsets[-1]) + 1 + TARGET_FRAME_OFFSET
        self._validate_arrays(keys)
        self.window_starts = self._build_window_starts()
        if not self.window_starts.size:
            raise ValueError(f"No complete windows were created from {self.npz_path}")

    def _validate_arrays(self, keys: Sequence[str]) -> None:
        n_rows = len(self.vision_feat)
        if n_rows == 0:
            raise ValueError(f"Dataset is empty: {self.npz_path}")
        for key in keys:
            if len(getattr(self, key)) != n_rows:
                raise ValueError(f"{key} has a different row count in {self.npz_path}")
        if self.vision_feat.shape != (n_rows, VISION_FEATURE_DIM):
            raise ValueError(f"Unexpected vision shape: {self.vision_feat.shape}")
        if self.gesture_seq.shape != (n_rows, SOURCE_GESTURE_SEQUENCE_LENGTH):
            raise ValueError(f"Unexpected gesture shape: {self.gesture_seq.shape}")
        observed = set(np.unique(self.gesture_seq).tolist())
        expected = set(LABEL_VOCAB.values()) | {MASK_LABEL_ID}
        if not observed <= expected:
            raise ValueError(f"Unexpected gesture IDs: {sorted(observed - expected)}")
        scores = np.asarray(self.ue_fma_score)
        if not np.isfinite(scores).all():
            raise ValueError("UE-FMA targets must be finite")
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        if score_min < UEFMA_MIN or score_max > UEFMA_MAX:
            raise ValueError("UE-FMA targets must be in [0,66]")
        self.clip_starts = np.flatnonzero(np.asarray(self.frame_index) == 0).astype(np.int64)
        self.clip_ends = np.r_[self.clip_starts[1:], n_rows].astype(np.int64)
        if not self.clip_starts.size or self.clip_starts[0] != 0:
            raise ValueError("The first merged-data clip must start at frame zero")
        for start, end in zip(self.clip_starts, self.clip_ends):
            length = int(end - start)
            frames = np.asarray(self.frame_index[start:end])
            if not np.array_equal(frames, np.arange(length, dtype=np.int64)):
                raise ValueError(f"Non-consecutive frames in clip {self.clip_id[start]}")
            invariant_fields = ["clip_id", "subject_id", "task_name"]
            if self.conditioning != CONDITIONING_NONE:
                invariant_fields.extend(PATIENT_NPZ_KEYS)
            for field in invariant_fields:
                values = np.asarray(getattr(self, field)[start:end])
                if np.any(values != values[0]):
                    raise ValueError(f"{field} changes within clip at row {start}")

    def _build_window_starts(self) -> np.ndarray:
        available = frozenset(
            canonical_subject_id(self.subject_id[start]) for start in self.clip_starts
        )
        if self.include_subjects is not None and not self.include_subjects <= available:
            raise ValueError(f"Missing subjects: {sorted(self.include_subjects - available)}")
        chunks: list[np.ndarray] = []
        selected: set[str] = set()
        for start, end in zip(self.clip_starts, self.clip_ends):
            subject = canonical_subject_id(self.subject_id[start])
            if self.include_subjects is not None and subject not in self.include_subjects:
                continue
            if int(end - start) < self.required_span:
                continue
            selected.add(subject)
            chunks.append(
                np.arange(
                    int(start),
                    int(end) - self.required_span + 1,
                    WINDOW_STRIDE,
                    dtype=np.int64,
                )
            )
        if self.include_subjects is not None and selected != set(self.include_subjects):
            missing = sorted(set(self.include_subjects) - selected)
            raise ValueError(f"Subjects have no complete temporal windows: {missing}")
        self.subjects = frozenset(selected)
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def select_training_clips(self, percentage: float, seed: int) -> dict[str, Any]:
        """Keep a fixed per-subject sample of complete clips and all their windows."""
        if not 0 < percentage <= 100:
            raise ValueError("clip percentage must be greater than zero and at most 100")
        generator = random.Random(seed)
        eligible: dict[str, list[tuple[str, str, int, int]]] = {}
        for start, end in zip(self.clip_starts, self.clip_ends):
            subject = canonical_subject_id(self.subject_id[start])
            if subject in self.subjects and int(end - start) >= self.required_span:
                eligible.setdefault(subject, []).append(
                    (str(self.clip_id[start]), str(self.task_name[start]), int(start), int(end))
                )

        def window_count(records: list[tuple[str, str, int, int]]) -> int:
            return sum((end - start - self.required_span) // WINDOW_STRIDE + 1
                       for _, _, start, end in records)

        selected_starts: list[int] = []
        selected_clips: list[dict[str, Any]] = []
        subject_counts: list[dict[str, Any]] = []
        for subject in sorted(eligible):
            clips = sorted(eligible[subject])
            count = max(1, ceil(len(clips) * percentage / 100.0))
            chosen = clips if percentage == 100 else sorted(generator.sample(clips, count))
            selected_starts.extend(start for _, _, start, _ in chosen)
            selected_clips.extend(
                {"subject_id": subject, "clip_id": clip, "task_name": task}
                for clip, task, _, _ in chosen
            )
            subject_counts.append({
                "subject_id": subject,
                "eligible_clips": len(clips), "selected_clips": len(chosen),
                "eligible_windows": window_count(clips),
                "selected_windows": window_count(chosen),
                "actual_clip_percentage": 100.0 * len(chosen) / len(clips),
            })
        eligible_windows = len(self.window_starts)
        if percentage < 100:
            window_clips = self.clip_starts[
                np.searchsorted(self.clip_starts, self.window_starts, side="right") - 1
            ]
            self.window_starts = self.window_starts[np.isin(window_clips, selected_starts)]
        eligible_count = sum(len(clips) for clips in eligible.values())
        selection = {
            "requested_clip_percentage": percentage,
            "seed": seed,
            "eligible_clips": eligible_count,
            "selected_clips": len(selected_clips),
            "eligible_windows": eligible_windows,
            "selected_windows": len(self.window_starts),
            "actual_clip_percentage": 100.0 * len(selected_clips) / eligible_count,
            "clips": selected_clips,
            "subjects": subject_counts,
        }
        self.training_selection = selection
        return selection

    def _window_indices(self, index: int) -> tuple[np.ndarray, int]:
        context = int(self.window_starts[index]) + self.context_offsets
        return context, int(context[-1]) + TARGET_FRAME_OFFSET

    def _prompt_text(self, task_name: str, target: int) -> str:
        """Build the prompt for one window (the text variant adds the patient sentence)."""
        if task_name not in TASK_DESCRIPTIONS:
            raise KeyError(f"No task description for {task_name!r}")
        description = TASK_DESCRIPTIONS[task_name]
        if not self.use_patient_text:
            return f"{description}\n{PREDICTION_PROMPT}"
        patient_text = (
            f"Patient information: paretic side is {self.paretic_side[target]}; "
            f"gender is {self.gender[target]}; age is {float(self.age[target]):.0f} years; "
            f"race is {self.race[target]}; time since stroke is "
            f"{float(self.time_since_stroke[target]):.1f}."
        )
        return f"{description}\n{patient_text}\n{PREDICTION_PROMPT}"

    def __len__(self) -> int:
        return int(len(self.window_starts))

    def __getitem__(self, index: int) -> dict[str, Any]:
        context, target = self._window_indices(index)
        task_name = str(self.task_name[target])
        sample: dict[str, Any] = {
            "text": self._prompt_text(task_name, target),
            "vision_feat": torch.from_numpy(
                np.array(self.vision_feat[context], dtype=np.float32, copy=True, order="C")
            ),
            "gesture_labels": torch.from_numpy(
                np.array(
                    self.gesture_seq[target, :TARGET_WINDOW_SIZE],
                    dtype=np.int64,
                    copy=True,
                    order="C",
                )
            ),
            "ue_fma_score": torch.tensor(float(self.ue_fma_score[target]), dtype=torch.float32),
            "subject_id": canonical_subject_id(self.subject_id[target]),
            "clip_id": str(self.clip_id[target]),
            "task_name": task_name,
            "target_frame_index": int(self.frame_index[target]),
        }
        if self.use_patient_prefix:
            assert self.patient_normalizer is not None
            encoded = self.patient_normalizer.encode(
                self.paretic_side[target],
                self.gender[target],
                self.age[target],
                self.race[target],
                self.time_since_stroke[target],
            )
            sample["patient_features"] = torch.from_numpy(encoded.copy())
        return sample


class StrokeRehabCollator:
    def __init__(self, tokenizer: Any, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty sample list")
        tokens = self.tokenizer(
            [str(sample["text"]) for sample in samples],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        batch: dict[str, Any] = {
            "input_ids": tokens["input_ids"],
            "text_attention_mask": tokens["attention_mask"].bool(),
            "vision_feat": torch.stack([sample["vision_feat"] for sample in samples]),
            "gesture_labels": torch.stack([sample["gesture_labels"] for sample in samples]),
            "ue_fma_score": torch.stack([sample["ue_fma_score"] for sample in samples]),
        }
        has_patient = ["patient_features" in sample for sample in samples]
        if any(has_patient) and not all(has_patient):
            raise ValueError("Cannot mix conditioned and unconditioned samples")
        if all(has_patient):
            batch["patient_features"] = torch.stack(
                [sample["patient_features"] for sample in samples]
            )
        for key in ("subject_id", "clip_id", "task_name", "target_frame_index"):
            batch[key] = [sample[key] for sample in samples]
        return batch


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: StrokeRehabDataset,
    collator: StrokeRehabCollator,
    config: RuntimeConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "collate_fn": collator,
    }
    if config.num_workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def validate_data_files(config: RuntimeConfig) -> None:
    for path in (config.train_data_path, config.test_data_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Set --data-root, or --raw-dir to build it from the raw data."
            )
    train_count = len(npz_subjects(config.train_data_path, config.cache_root))
    test_count = len(npz_subjects(config.test_data_path, config.cache_root))
    if train_count != EXPECTED_TRAIN_SUBJECTS or test_count != EXPECTED_TEST_SUBJECTS:
        raise ValueError(
            f"Expected 32 train/8 test subjects; found {train_count} train/{test_count} test"
        )
