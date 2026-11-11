"""Exercise data for the multi-agent system, processed exactly as in training.

This module reads the raw per-clip files in `data/stroke_rehab` and, with training's own
converter (`train/raw_data.py`), creates a private, content-addressed cache with the keys
consumed by `StrokeRehabDataset`. The cached data
then goes through `StrokeRehabDataset` itself, so the 24-frame context, the
frame skip and stride, the prompt text, the patient encoding and the five target
positions (frames t..t+4) are the training code's, not a re-implementation.

Each clip becomes one *exercise* of a patient. Ground truth (the primitive labels and
UE-FMA) is reachable only through `ExerciseSource.labels` and `true_uefma`, which the
evaluator uses after a run; agents never call them.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from multi_agents.settings import CACHE_DIR, HORIZONS
from multi_agents.state import PatientProfile
from multi_agents.train_bridge import (
    PATIENT_NPZ_KEYS,
    RawClip,
    StrokeRehabDataset,
    canonical_subject_id,
    discover_raw_clips,
    extract_npz_members,
    parse_clip_id,
    subjects_by_set,
    train_config,
    write_npz,
)

SPLITS = ("test", "validation", "train")

# --------------------------------------------------------------------------- raw data


def split_subjects(raw_dir: Path, split: str) -> tuple[str, ...]:
    """Subjects of a split, read from each subject's `training_test_set` field.

    `validation` is training's fixed full-run validation set and `train` is the rest
    of the training split, i.e. the subjects the full-run checkpoint was fit on.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    by_set = subjects_by_set(raw_dir)
    validation = set(train_config.FULL_VALIDATION_SUBJECTS)
    if split == "test":
        return tuple(by_set["test"])
    if split == "validation":
        return tuple(sorted(validation & set(by_set["training"])))
    return tuple(sorted(set(by_set["training"]) - validation))


def build_npz_from_raw(raw_dir: Path, clips: Sequence[RawClip], tag: str) -> Path:
    """The clips as a training-format NPZ (`train/raw_data.py`), cached once.

    The file name hashes the clip list, so the memmap cache `StrokeRehabDataset` builds
    from it can never go stale.
    """
    digest = hashlib.sha256("\n".join(sorted(c.clip_id for c in clips)).encode()).hexdigest()[:12]
    out_path = CACHE_DIR / "npz" / f"raw_{tag}_{digest}.npz"
    if out_path.is_file():
        return out_path
    return write_npz(raw_dir, clips, out_path)


# ------------------------------------------------------------------------- exercises


@dataclass(frozen=True)
class Exercise:
    subject_id: str
    clip_id: str
    task_name: str
    repetition: int
    trial: int
    window_indices: tuple[int, ...]
    clip_start: int
    n_frames: int

    @property
    def key(self) -> str:
        return f"{self.subject_id}/{self.clip_id}"

    @property
    def n_windows(self) -> int:
        return len(self.window_indices)


class ExerciseSource:
    """Groups `StrokeRehabDataset` windows into per-patient, ordered exercises."""

    def __init__(
        self,
        npz_path: Path,
        subjects: Iterable[str] | None,
        conditioning: str,
        normalizer=None,
        cache_root: Path = CACHE_DIR / "memmap",
    ) -> None:
        self.npz_path = Path(npz_path)
        self.cache_root = Path(cache_root)
        self.dataset = StrokeRehabDataset(
            self.npz_path,
            self.cache_root,
            include_subjects=subjects,
            conditioning=conditioning,
            patient_normalizer=normalizer,
        )
        self._target_offset = (
            (train_config.CONTEXT_WINDOW_SIZE - 1) * train_config.FRAME_SKIP
            + train_config.TARGET_FRAME_OFFSET
        )
        self._by_subject: dict[str, list[Exercise]] = defaultdict(list)
        self._by_clip: dict[str, Exercise] = {}
        dataset = self.dataset
        clip_positions = np.searchsorted(dataset.clip_starts, dataset.window_starts, side="right") - 1
        grouped: dict[int, list[int]] = defaultdict(list)
        for window_index, clip_position in enumerate(clip_positions):
            grouped[int(clip_position)].append(window_index)
        for clip_position, window_indices in grouped.items():
            start = int(dataset.clip_starts[clip_position])
            clip_id = str(dataset.clip_id[start])
            try:
                name = parse_clip_id(clip_id)
                repetition, trial = name.repetition, name.trial
            except ValueError:
                repetition = trial = 0
            exercise = Exercise(
                subject_id=canonical_subject_id(dataset.subject_id[start]),
                clip_id=clip_id,
                task_name=str(dataset.task_name[start]),
                repetition=repetition,
                trial=trial,
                window_indices=tuple(sorted(window_indices)),
                clip_start=start,
                n_frames=int(dataset.clip_ends[clip_position] - start),
            )
            self._by_subject[exercise.subject_id].append(exercise)
            self._by_clip[exercise.key] = exercise
        self._patient_arrays: dict[str, np.ndarray] | None = None

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_subject))

    def find(self, clip_id: str, subject_id: str | None = None) -> Exercise:
        """Look up an exercise; the subject disambiguates clip ids shared across subjects."""
        key = f"{subject_id}/{clip_id}" if subject_id else None
        if key is None:
            matches = [e for e in self._by_clip.values() if e.clip_id == clip_id]
            if len(matches) > 1:
                raise KeyError(f"{clip_id} exists for several subjects; pass subject_id")
            key = matches[0].key if matches else ""
        if key not in self._by_clip:
            raise KeyError(f"{clip_id} has no complete {train_config.CONTEXT_WINDOW_SIZE}-frame window")
        return self._by_clip[key]

    def exercises(self, subject_id: str) -> list[Exercise]:
        """A patient's exercises in pseudo-longitudinal (session) order.

        The data has no session timestamps, so repetition r of every task is treated as
        session r, with the tasks in protocol order within it, then trial and clip id.
        """
        task_rank = {task: rank for rank, task in enumerate(train_config.TASKS)}
        return sorted(
            self._by_subject[subject_id],
            key=lambda e: (e.repetition, task_rank.get(e.task_name, 99), e.trial, e.clip_id),
        )

    def window_samples(self, exercise: Exercise) -> list[dict]:
        """Training-format samples (text, vision_feat, optional patient_features)."""
        samples = []
        for window_index in exercise.window_indices:
            sample = self.dataset[window_index]
            sample.pop("gesture_labels", None)
            sample.pop("ue_fma_score", None)
            samples.append(sample)
        return samples

    def frames(self, exercise: Exercise) -> Iterator[np.ndarray]:
        """The clip's feature frames in time order, as a live sensor would deliver them."""
        for frame_index in range(exercise.n_frames):
            yield np.array(self.dataset.vision_feat[exercise.clip_start + frame_index], dtype=np.float32)

    def stream_context(self, exercise: Exercise) -> dict:
        """Per-clip model inputs that do not change frame to frame (prompt, patient prefix).

        Taken from the dataset's own first window, so streamed windows get exactly the
        text and patient features the offline windows get.
        """
        sample = self.dataset[exercise.window_indices[0]]
        return {"text": str(sample["text"]), "patient_features": sample.get("patient_features")}

    def prompt(self, exercise: Exercise) -> str:
        """The exact text the model receives (task, patient sentence if text-conditioned)."""
        return self.stream_context(exercise)["text"]

    def target_frames(self, exercise: Exercise) -> np.ndarray:
        starts = self.dataset.window_starts[list(exercise.window_indices)]
        return (starts + self._target_offset - exercise.clip_start).astype(np.int64)

    def profile(self, subject_id: str) -> PatientProfile:
        row = self._by_subject[subject_id][0].clip_start
        arrays = self._load_patient_arrays()
        if arrays is None:
            return PatientProfile(subject_id=subject_id)
        return PatientProfile(
            subject_id=subject_id,
            paretic_side=str(arrays["paretic_side"][row]),
            gender=str(arrays["gender"][row]),
            age=float(arrays["age"][row]),
            race=str(arrays["race"][row]),
            time_since_stroke=float(arrays["time_since_stroke"][row]),
        )

    # Ground truth: evaluator only.
    def labels(self, exercise: Exercise) -> np.ndarray:
        targets = self.target_frames(exercise) + exercise.clip_start
        return np.asarray(self.dataset.gesture_seq[targets, :HORIZONS], dtype=np.int64)

    def true_uefma(self, subject_id: str) -> float:
        return float(self.dataset.ue_fma_score[self._by_subject[subject_id][0].clip_start])

    def _load_patient_arrays(self) -> dict[str, np.ndarray] | None:
        if self._patient_arrays is None:
            try:
                paths = extract_npz_members(self.npz_path, self.cache_root, PATIENT_NPZ_KEYS)
            except KeyError:
                return None
            self._patient_arrays = {
                key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in paths.items()
            }
        return self._patient_arrays


def open_source(
    *,
    split: str,
    raw_dir: Path,
    conditioning: str,
    normalizer=None,
    subjects: Sequence[str] | None = None,
) -> ExerciseSource:
    """Build the exercise source the entry points share."""
    selected = tuple(subjects) if subjects else split_subjects(raw_dir, split)
    cache_path = build_npz_from_raw(raw_dir, discover_raw_clips(raw_dir, selected), split)
    return ExerciseSource(cache_path, selected, conditioning, normalizer)
