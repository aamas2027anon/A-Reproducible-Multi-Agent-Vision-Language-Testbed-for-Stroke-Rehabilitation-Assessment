"""Population priors for the planning and safety agents.

Fitted from ground-truth labels of the subjects the full-run checkpoint was trained
on (the training split minus the four validation subjects), never from validation
or test subjects. Frame-level statistics are used because the model's five target
positions are consecutive frames t..t+4 and consecutive windows tile the timeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from multi_agents.settings import CACHE_DIR, NUM_PRIMITIVES, PRIMITIVES
from multi_agents.train_bridge import train_config


@dataclass(frozen=True)
class PopulationPriors:
    transition: np.ndarray
    class_prior: np.ndarray
    subjects: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "classes": list(PRIMITIVES),
            "transition": self.transition.tolist(),
            "class_prior": self.class_prior.tolist(),
            "subjects": list(self.subjects),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "PopulationPriors":
        # Other keys (such as the per-task class priors older caches stored) are ignored.
        return cls(
            transition=np.asarray(payload["transition"], dtype=np.float64),
            class_prior=np.asarray(payload["class_prior"], dtype=np.float64),
            subjects=tuple(payload["subjects"]),
        )


def fit_priors(sequences: Iterable[tuple[str, np.ndarray]], subjects: Iterable[str], smoothing: float = 1.0) -> PopulationPriors:
    transitions = np.full((NUM_PRIMITIVES, NUM_PRIMITIVES), smoothing)
    classes = np.full(NUM_PRIMITIVES, smoothing)
    for _task_name, labels in sequences:
        labels = labels[labels < NUM_PRIMITIVES]
        if labels.size == 0:
            continue
        counts = np.bincount(labels, minlength=NUM_PRIMITIVES)
        classes += counts
        np.add.at(transitions, (labels[:-1], labels[1:]), 1.0)
    return PopulationPriors(
        transition=transitions / transitions.sum(axis=1, keepdims=True),
        class_prior=classes / classes.sum(),
        subjects=tuple(sorted(subjects)),
    )


def raw_sequences(raw_dir: Path, subjects: Iterable[str]) -> Iterator[tuple[str, np.ndarray]]:
    from multi_agents.data import discover_raw_clips, parse_clip_id

    vocabulary = train_config.LABEL_VOCAB
    for clip in discover_raw_clips(raw_dir, subjects):
        names = [line.strip() for line in clip.label_path.read_text().splitlines() if line.strip()]
        yield parse_clip_id(clip.clip_id).task_name, np.asarray([vocabulary[n] for n in names], dtype=np.int64)


def load_or_fit_priors(subjects: Iterable[str], raw_dir: Path) -> PopulationPriors:
    """Fit once per subject set and cache the result as JSON."""
    subjects = tuple(sorted(subjects))
    digest = hashlib.sha256(",".join(subjects).encode()).hexdigest()[:12]
    path = CACHE_DIR / f"priors_{digest}.json"
    if path.is_file():
        return PopulationPriors.from_json(json.loads(path.read_text(encoding="utf-8")))
    if not Path(raw_dir, "video_extracted_features").is_dir():
        raise FileNotFoundError(f"Raw StrokeRehab features not found under {raw_dir}")
    priors = fit_priors(raw_sequences(Path(raw_dir), subjects), subjects)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(priors.to_json(), indent=2), encoding="utf-8")
    return priors


def resolve_priors(raw_dir: Path) -> PopulationPriors:
    """Priors from the subjects the full-run checkpoint trained on.

    That is the training split minus the four validation subjects, read from raw metadata.
    """
    if not Path(raw_dir, "kinematics").is_dir():
        raise FileNotFoundError(f"Raw StrokeRehab metadata not found under {raw_dir}")
    from multi_agents.data import split_subjects

    subjects = split_subjects(Path(raw_dir), "train")
    return load_or_fit_priors(subjects, raw_dir)
