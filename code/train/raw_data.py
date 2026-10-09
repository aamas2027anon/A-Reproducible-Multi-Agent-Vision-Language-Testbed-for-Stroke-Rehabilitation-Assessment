"""Raw StrokeRehab clips -> the merged NPZs that `StrokeRehabDataset` reads.

The raw tree has `video_extracted_features/<subject>/{features,groundtruth}/<clip>.*`
and `kinematics/<subject>/metadata.xml`. Training and the multi-agent system both go
through `write_npz`, so they see the same frames, labels and patient fields.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from config import LABEL_VOCAB, MASK_LABEL_ID, SOURCE_GESTURE_SEQUENCE_LENGTH, VISION_FEATURE_DIM
from data import canonical_subject_id

TRAIN_NPZ_NAME = "stroke_rehab_train_all_tasks.npz"
TEST_NPZ_NAME = "stroke_rehab_test_all_tasks.npz"

# Raw clip names use a few spellings the training task vocabulary does not.
_TASK_PREFIXES = (
    ("rtt", "tabletop"),
    ("tabletop", "tabletop"),
    ("shelf", "shelf"),
    ("face_wash", "face_wash"),
    ("facewash", "face_wash"),
    ("deodrant", "deodorant"),
    ("deodorant", "deodorant"),
    ("feeding", "feeding"),
    ("drinking", "drinking"),
    ("combing", "combing"),
    ("glasses", "glasses"),
    ("brushing", "brushing"),
)
_METADATA_FIELDS = ("paretic_side", "gender", "age", "race", "time_since_stroke", "UE_FMA_score")


@dataclass(frozen=True)
class ClipName:
    task_name: str
    repetition: int
    trial: int


def parse_clip_id(clip_id: str) -> ClipName:
    """`s05_RTT_left_side2_1` -> tabletop, repetition 2, trial 1."""
    text = re.sub(r"^s\d+_", "", clip_id, flags=re.IGNORECASE)
    trial = 0
    match = re.search(r"_(\d+)$", text)
    if match:
        trial, text = int(match.group(1)), text[: match.start()]
    repetition = 0
    match = re.search(r"(\d+)$", text)
    if match:
        repetition, text = int(match.group(1)), text[: match.start()]
    lowered = text.lower()
    for prefix, task in _TASK_PREFIXES:
        if lowered.startswith(prefix):
            return ClipName(task, repetition, trial)
    raise ValueError(f"Cannot map clip {clip_id!r} to a training task")


@dataclass(frozen=True)
class RawClip:
    subject_id: str
    clip_id: str
    feature_path: Path
    label_path: Path


def read_metadata(raw_dir: Path, subject_id: str) -> dict[str, str] | None:
    path = Path(raw_dir) / "kinematics" / subject_id / "metadata.xml"
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    return {child.tag: str(child.attrib.get("value")) for child in root}


def subjects_by_set(raw_dir: Path) -> dict[str, list[str]]:
    """Subjects grouped by their metadata `training_test_set` (training, test, none)."""
    by_set: dict[str, list[str]] = defaultdict(list)
    for subject_dir in sorted((Path(raw_dir) / "video_extracted_features").glob("s*")):
        metadata = read_metadata(raw_dir, subject_dir.name)
        if metadata is None:
            continue
        by_set[metadata.get("training_test_set", "none")].append(canonical_subject_id(subject_dir.name))
    return by_set


def discover_raw_clips(
    raw_dir: Path, subjects: Iterable[str], clip_ids: Iterable[str] | None = None
) -> list[RawClip]:
    wanted = None if clip_ids is None else set(clip_ids)
    clips: list[RawClip] = []
    for subject in sorted(subjects):
        folder = Path(raw_dir) / "video_extracted_features" / subject
        for feature_path in sorted((folder / "features").glob("*.npy")):
            if wanted is not None and feature_path.stem not in wanted:
                continue
            label_path = folder / "groundtruth" / f"{feature_path.stem}.txt"
            if not label_path.is_file():
                continue
            clips.append(RawClip(subject, feature_path.stem, feature_path, label_path))
    if wanted is not None and {clip.clip_id for clip in clips} != wanted:
        missing = sorted(wanted - {clip.clip_id for clip in clips})
        raise FileNotFoundError(f"Raw clips not found: {missing}")
    if not clips:
        raise FileNotFoundError(f"No raw clips with ground truth found under {raw_dir}")
    return clips


def write_npz(raw_dir: Path, clips: Sequence[RawClip], out_path: Path) -> Path:
    """Write the clips as one NPZ with the training pipeline's keys and semantics.

    `gesture_seq[i, h]` is the label of frame i+h (the training targets t..t+4),
    with `MASK_LABEL_ID` past the end of the clip.
    """
    columns: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata_cache: dict[str, dict[str, str]] = {}
    for clip in clips:
        if clip.subject_id not in metadata_cache:
            metadata = read_metadata(raw_dir, clip.subject_id)
            if metadata is None or any(field not in metadata for field in _METADATA_FIELDS):
                raise ValueError(f"{clip.subject_id} has no complete metadata.xml")
            metadata_cache[clip.subject_id] = metadata
        metadata = metadata_cache[clip.subject_id]
        features = np.load(clip.feature_path).astype(np.float32, copy=False)
        if features.ndim != 2 or VISION_FEATURE_DIM not in features.shape:
            raise ValueError(f"Unexpected feature shape {features.shape} in {clip.feature_path}")
        if features.shape[0] == VISION_FEATURE_DIM:
            features = features.T
        names = [line.strip() for line in clip.label_path.read_text().splitlines() if line.strip()]
        if len(names) != len(features):
            raise ValueError(f"{clip.clip_id}: {len(names)} labels for {len(features)} frames")
        labels = np.asarray([LABEL_VOCAB[name] for name in names], dtype=np.int64)
        length = len(labels)
        offsets = np.arange(length)[:, None] + np.arange(SOURCE_GESTURE_SEQUENCE_LENGTH)[None, :]
        gesture_seq = np.where(offsets < length, labels[np.minimum(offsets, length - 1)], MASK_LABEL_ID)
        repeated = lambda value: np.full(length, value)  # noqa: E731
        columns["vision_feat"].append(np.ascontiguousarray(features))
        columns["gesture_seq"].append(gesture_seq.astype(np.int64))
        columns["frame_index"].append(np.arange(length, dtype=np.int64))
        columns["subject_id"].append(repeated(clip.subject_id))
        columns["clip_id"].append(repeated(clip.clip_id))
        columns["task_name"].append(repeated(parse_clip_id(clip.clip_id).task_name))
        columns["UE_FMA_score"].append(repeated(float(metadata["UE_FMA_score"])).astype(np.float32))
        for field in ("paretic_side", "gender", "race"):
            columns[field].append(repeated(metadata[field]))
        for field in ("age", "time_since_stroke"):
            columns[field].append(repeated(float(metadata[field])))

    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.stem}.{os.getpid()}.partial.npz")
    np.savez(temporary, **arrays)
    temporary.replace(out_path)
    out_path.with_suffix(".json").write_text(
        json.dumps({"clips": [c.clip_id for c in clips], "rows": int(len(arrays["frame_index"]))}, indent=2),
        encoding="utf-8",
    )
    return out_path


def build_training_npzs(raw_dir: Path, data_root: Path) -> tuple[Path, Path]:
    """Write the train (training set, 32 subjects) and test (8 subjects) NPZs if missing."""
    raw_dir, data_root = Path(raw_dir), Path(data_root)
    if not (raw_dir / "video_extracted_features").is_dir() or not (raw_dir / "kinematics").is_dir():
        raise FileNotFoundError(
            f"Raw StrokeRehab data not found under {raw_dir} (expected video_extracted_features/ "
            "and kinematics/). Place or symlink it there, set data_dir in project_config.toml, "
            "or pass --raw-dir"
        )
    by_set = subjects_by_set(raw_dir)
    outputs = []
    for name, subject_set in ((TRAIN_NPZ_NAME, "training"), (TEST_NPZ_NAME, "test")):
        path = data_root / name
        if not path.is_file():
            print(f"Building {path} from the raw {subject_set} subjects in {raw_dir} ...")
            write_npz(raw_dir, discover_raw_clips(raw_dir, by_set[subject_set]), path)
        outputs.append(path)
    return outputs[0], outputs[1]
