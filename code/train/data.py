from __future__ import annotations

import json
import os
import random
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from config import (
    CLASS_NAMES,
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
    N_FOLDS,
    NUM_PRIMITIVE_CLASSES,
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


def create_fold_assignments(config: RuntimeConfig) -> pd.DataFrame:
    subjects = np.asarray(npz_subjects(config.train_data_path, config.cache_root))
    if len(subjects) != EXPECTED_TRAIN_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_SUBJECTS} train subjects, found {len(subjects)}"
        )
    result = build_fold_assignments(subjects, config.seed, config.n_folds)

    config.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_root / "fold_assignments.csv"
    json_path = config.output_root / "fold_assignments.json"
    payload = {
        "seed": config.seed,
        "n_folds": config.n_folds,
        "fold_sizes": {
            str(int(fold_id)): int(size)
            for fold_id, size in result.groupby("fold_id").size().items()
        },
        "assignments": result.to_dict("records"),
    }

    # Seed and fold count are tunable. Persist the requested assignment instead
    # of rejecting a run merely because this output root was used previously
    # with different values. Both files are replaced atomically so readers never
    # observe a partially written artifact.
    temporary_csv = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.partial")
    temporary_json = json_path.with_name(f".{json_path.name}.{os.getpid()}.partial")
    try:
        result.to_csv(temporary_csv, index=False)
        temporary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_csv.replace(csv_path)
        temporary_json.replace(json_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    return result


def build_fold_assignments(
    subjects: Sequence[str], seed: int, n_folds: int = N_FOLDS
) -> pd.DataFrame:
    canonical = np.asarray(sorted(canonical_subject_id(value) for value in subjects))
    if len(canonical) != EXPECTED_TRAIN_SUBJECTS or len(np.unique(canonical)) != len(canonical):
        raise ValueError("Fold construction requires exactly 32 unique subjects")
    rows: list[dict[str, Any]] = []
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (train_indices, holdout_indices) in enumerate(splitter.split(canonical)):
        train_subjects = tuple(sorted(canonical[train_indices].tolist()))
        holdout_subjects = tuple(sorted(canonical[holdout_indices].tolist()))
        if set(train_subjects) & set(holdout_subjects):
            raise AssertionError(f"Fold {fold_id} train/holdout overlap")
        rows.extend(
            {
                "subject_id": subject,
                "fold_id": fold_id,
                "fold_seed": seed + fold_id,
            }
            for subject in holdout_subjects
        )
    result = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    if len(result) != EXPECTED_TRAIN_SUBJECTS or not result["subject_id"].is_unique:
        raise AssertionError("Fold assignments do not cover 32 subjects exactly once")
    fold_sizes = result.groupby("fold_id").size()
    if len(fold_sizes) != n_folds or fold_sizes.max() - fold_sizes.min() > 1:
        raise AssertionError("Cross-validation folds must be balanced")

    return result


def fold_subjects(assignments: pd.DataFrame, fold_id: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = tuple(sorted(int(value) for value in assignments["fold_id"].unique()))
    if fold_id not in available:
        raise ValueError(f"Invalid fold {fold_id}; expected one of {available}")
    holdout = tuple(sorted(assignments.loc[assignments["fold_id"] == fold_id, "subject_id"]))
    train = tuple(sorted(set(assignments["subject_id"]) - set(holdout)))
    if len(train) + len(holdout) != EXPECTED_TRAIN_SUBJECTS:
        raise AssertionError("Unexpected cross-validation fold sizes")
    return train, holdout


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
