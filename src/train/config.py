"""Settings of the testbed: values from config.yaml, plus the fixed data and model layout."""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

# config.yaml next to code/train/, or the file named by STROKE_REHAB_CONFIG.
CONFIG_FILE = Path(
    os.environ.get("STROKE_REHAB_CONFIG", Path(__file__).resolve().parents[1] / "config.yaml")
).expanduser().absolute()
with CONFIG_FILE.open(encoding="utf-8") as _handle:
    CONFIG: dict[str, Any] = yaml.safe_load(_handle)
_TRAINING = CONFIG["training"]
_LORA = CONFIG["lora"]

MODEL_NAME = CONFIG["model"]["name"]

FRAME_SKIP = 5
CONTEXT_WINDOW_SIZE = 24
WINDOW_STRIDE = FRAME_SKIP
TARGET_WINDOW_SIZE = 5
SOURCE_GESTURE_SEQUENCE_LENGTH = 5
TARGET_FRAME_OFFSET = 0
VISION_FEATURE_DIM = 432
QWEN_HIDDEN_SIZE = 1024
PATIENT_FEATURE_DIM = 12
PATIENT_PREFIX_DIM = PATIENT_FEATURE_DIM
CONDITIONED_VISION_DIM = QWEN_HIDDEN_SIZE - PATIENT_PREFIX_DIM
CONDITIONING_MODE = "per_frame_patient_prefix_12_plus_visual_1012"
MAX_TEXT_LENGTH = int(CONFIG["model"]["max_text_length"])
PREDICTION_PROMPT = "Predict five movement primitives and one UE-FMA score."

NUM_EPOCHS = int(_TRAINING["epochs"])
PER_DEVICE_BATCH_SIZE = int(_TRAINING["batch_size"])
GRADIENT_ACCUMULATION_STEPS = int(_TRAINING["gradient_accumulation"])
NUM_WORKERS = int(_TRAINING["num_workers"])
LEARNING_RATE = float(_TRAINING["learning_rate"])
WEIGHT_DECAY = float(_TRAINING["weight_decay"])
SCHEDULER_GAMMA = float(_TRAINING["scheduler_gamma"])
GRAD_CLIP_NORM = float(_TRAINING["gradient_clip_norm"])
UEFMA_LOSS_WEIGHT = float(_TRAINING["ue_fma_loss_weight"])
ENABLE_EARLY_STOPPING = bool(_TRAINING["early_stopping"]["enabled"])
EARLY_STOPPING_PATIENCE = int(_TRAINING["early_stopping"]["patience"])
EARLY_STOPPING_MIN_DELTA = float(_TRAINING["early_stopping"]["min_delta"])
AUTO_RESUME = bool(_TRAINING["resume"])
USE_GRADIENT_CHECKPOINTING = bool(_TRAINING["gradient_checkpointing"])
DETERMINISTIC_TRAINING = bool(_TRAINING["deterministic"])
SMOKE_BATCHES = int(_TRAINING["smoke_batches"])

LORA_R = int(_LORA["r"])
LORA_ALPHA = int(_LORA["alpha"])
LORA_DROPOUT = float(_LORA["dropout"])
LORA_TARGET_MODULES = tuple(_LORA["target_modules"])

EXPECTED_TRAIN_SUBJECTS = 32
EXPECTED_TEST_SUBJECTS = 8
FULL_VALIDATION_SUBJECTS = frozenset(_TRAINING["validation_subjects"])
# Every seed trains all three variants, so they are paired by seed. The first one is also
# the seed of the significance tests in the reports.
def resolve_seeds(initial: tuple[int, ...], count: int) -> tuple[int, ...]:
    if not initial or len(set(initial)) != len(initial) or count <= 0:
        raise ValueError("training seeds must be nonempty and unique; num_seeds must be positive")
    result = list(initial)
    generator = random.Random(initial[0])
    used = set(result)
    while len(result) < count:
        candidate = generator.randrange(2**32)
        if candidate not in used:
            result.append(candidate)
            used.add(candidate)
    return tuple(result[:count])


_INITIAL_SEEDS = tuple(int(seed) for seed in _TRAINING["seeds"])
NUM_SEEDS = _TRAINING.get("num_seeds", len(_INITIAL_SEEDS))
if isinstance(NUM_SEEDS, bool) or not isinstance(NUM_SEEDS, int):
    raise ValueError("training.num_seeds must be a positive integer")
SEEDS = resolve_seeds(_INITIAL_SEEDS, NUM_SEEDS)
CLIP_PERCENTAGE = float(_TRAINING.get("clip_percentage", 100.0))
if not 0 < CLIP_PERCENTAGE <= 100:
    raise ValueError("training.clip_percentage must be greater than zero and at most 100")
SEED = SEEDS[0]
N_PERMUTATIONS = int(CONFIG["statistics"]["sign_flips"])

TASKS = (
    "tabletop",
    "shelf",
    "feeding",
    "drinking",
    "combing",
    "glasses",
    "deodorant",
    "face_wash",
    "brushing",
)
TASK_DESCRIPTIONS = {
    "shelf": (
        "Shelf task. On table, two Plexiglas shelves (length: 90 cm; width: 25.5 cm; "
        "height: 33, 53 cm) with three targets (22.5, 45, and 67.5 cm from left-most "
        "edge) placed 20 cm from table edge and center target (diameter: 5 cm) placed "
        "2.5 cm from table edge. Toilet paper roll wrapped in self-adhesive wrap. "
        "Move the toilet paper back and forth between the center target and each target "
        "on the shelf, returning the arm to rest between each motion and the end."
    ),
    "tabletop": (
        "Tabletop task. On table, horizontal circular array (diameter: 48.5 cm) of eight "
        "outer targets and center target (diameter: 5 cm) placed 2.5 cm from table edge. "
        "Toilet paper roll wrapped in self-adhesive wrap. Move the roll back and forth "
        "between the center and each outer target, returning the arm to rest between "
        "each motion and the end."
    ),
    "feeding": (
        "Feeding. On table, paper plate (diameter: 21.6 cm) placed at midline, 5 cm from "
        "table edge; utensils placed three cm from table edge and five cm from either "
        "side of plate; plastic sandwich baggie containing a slice of bread placed 25 cm "
        "from table edge, and 23 cm left of midline; and margarine packet placed 32 cm "
        "from table edge and 17 cm right of midline. Paper plate, fork, knife, re-sealable "
        "sandwich baggie, slice of bread, single-serve margarine packet. Pick up the "
        "sandwich baggie and open it, remove the bread and put it on the plate, open the "
        "margarine packet and spread margarine on the bread, cut the bread into four "
        "pieces, cut off and eat a small piece."
    ),
    "drinking": (
        "Drinking. On table, water bottle and paper cup placed 18 cm left and right of "
        "midline, respectively, and 25 cm from table edge. Water bottle (12 oz), cup "
        "(four oz). Pick up the water battle and open it, pour some water into the cup, "
        "take a sip of water, place the cup on the table, and replace the cap on the bottle."
    ),
    "combing": (
        "Combing hair. On table, comb placed at midline, 25 cm from table edge. Comb. "
        "Pick up the comb and comb both sides of the head."
    ),
    "glasses": (
        "Donning glasses. On table, pair of glasses placed at midline, 25 cm from table "
        "edge. Pair of glasses. Pick up the glasses, open the sides, put on the glasses, "
        "briefly place your hands down on the table, and then remove the glasses and "
        "place them on the table."
    ),
    "deodorant": (
        "Applying deodorant. On table, deodorant placed at midline, 25 cm from table edge. "
        "Deodorant (solid, twist-base). Pick up the deodorant, remove the cap, twist the "
        "base a few times, apply the deodorant to each armpit, replace the cap, untwist "
        "the base, place the deodorant on the table."
    ),
    "face_wash": (
        "Washing face. At counter, small tub (length: 32.3 cm; width: 24.1 cm; depth: "
        "2.5 cm) in sink; two folded washcloths on each side of counter next to sink, "
        "30 cm from counter edge. Washcloths, tub, faucet handles. Fill the tub with "
        "water, take one washcloth and dip it into water, wring it, wipe each side of "
        "the face, and place it back on the counter. Then use the other washcloth to "
        "dry the face, and place it back on the counter."
    ),
    "brushing": (
        "Brushing teeth. At counter, toothpaste and toothbrush placed on either side of "
        "the sink, 30 cm from counter edge. Travel-sized toothpaste, toothbrush with "
        "built-up foam grip, faucet handles. Wet the toothbrush, open the toothpaste and "
        "apply it to the toothbrush, replace the cap on the toothpaste tube, brush teeth, "
        "rinse the toothbrush and mouth, and place the toothbrush back on the counter."
    ),
}

LABEL_VOCAB = {
    "reach": 0,
    "reposition": 1,
    "rest": 2,
    "stabilize": 3,
    "transport": 4,
}
CLASS_NAMES = tuple(LABEL_VOCAB)
NUM_PRIMITIVE_CLASSES = len(CLASS_NAMES)
MASK_LABEL_ID = NUM_PRIMITIVE_CLASSES
POSITION_NAMES = ("current", "future_1", "future_2", "future_3", "future_4")
UEFMA_MIN, UEFMA_MAX = 0.0, 66.0

PARETIC_SIDE_CATEGORIES = ("left", "right")
GENDER_CATEGORIES = ("Female", "Male")
RACE_CATEGORIES = (
    "Asian",
    "Black/African-American",
    "More Than One Race",
    "Other",
    "Unknown/Not Reported",
    "White/Caucasian",
)

# How patient information reaches the model. "text" writes the patient facts into the
# prompt and is architecturally identical to "none"; only "prefix" changes the network,
# reserving PATIENT_PREFIX_DIM of the hidden width for a per-frame patient embedding.
CONDITIONING_NONE = "none"
CONDITIONING_TEXT = "text"
CONDITIONING_PREFIX = "prefix"
CONDITIONING_MODES = (CONDITIONING_NONE, CONDITIONING_TEXT, CONDITIONING_PREFIX)

VARIANTS = (
    ("qwen_without_patient_conditioning", CONDITIONING_NONE),
    ("qwen_with_patient_text", CONDITIONING_TEXT),
    ("qwen_with_patient_prefix", CONDITIONING_PREFIX),
)
BASE_RUN_NAME = VARIANTS[0][0]
# Heaviest variant first so an out-of-memory configuration fails on the first run of a
# job rather than after the cheaper ones have already spent GPU hours.
TRAINING_VARIANTS = (VARIANTS[2], VARIANTS[1], VARIANTS[0])


# Output directory of the multi-seed run, under TRAINING_OUTPUT_DIR.
EXPERIMENT_DIRNAME = "multiseed"


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path
    output_root: Path
    cache_root: Path
    model_name: str = MODEL_NAME
    seed: int = SEED
    batch_size: int = PER_DEVICE_BATCH_SIZE
    num_workers: int = NUM_WORKERS
    epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    resume: bool = AUTO_RESUME
    clip_percentage: float = CLIP_PERCENTAGE
    # Batches per epoch and per prediction pass; None runs everything. Set only by the
    # smoke run, which checks that the pipeline works, not what it scores.
    limit_batches: int | None = None

    @property
    def train_data_path(self) -> Path:
        return self.data_root / "stroke_rehab_train_all_tasks.npz"

    @property
    def test_data_path(self) -> Path:
        return self.data_root / "stroke_rehab_test_all_tasks.npz"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("data_root", "output_root", "cache_root"):
            result[key] = portable_path(result[key])
        if os.path.isabs(self.model_name):
            result["model_name"] = portable_path(self.model_name)
        return result


# Paths resolve against the directory of the config file.
PROJECT_ROOT = CONFIG_FILE.parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def project_path(key: str) -> Path:
    """A `paths` entry of the config file; relative values resolve against its directory."""
    return Path(os.path.normpath(PROJECT_ROOT / Path(str(CONFIG["paths"][key])).expanduser()))


def portable_path(path: Path | str) -> str:
    """Record paths relative to the config directory or the repository root.

    Preserve config-relative paths for existing runs; also retain paths to repository-
    level outputs and models. External locations are recorded by their name only.
    """
    path = Path(path).expanduser()
    for candidate, root in (
        (Path(os.path.normpath(path.absolute())), PROJECT_ROOT),
        (path.resolve(), PROJECT_ROOT.resolve()),
        (Path(os.path.normpath(path.absolute())), REPOSITORY_ROOT),
        (path.resolve(), REPOSITORY_ROOT.resolve()),
    ):
        if candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
    return f"<external>/{path.name}"


# The directory layout. The raw data and the downloaded models are read from where the
# config file points; everything a run writes is under OUTPUT_DIR.
RAW_DATA_DIR = project_path("data_dir")
MODELS_DIR = project_path("models_dir")
BASE_MODEL_DIR = MODELS_DIR / MODEL_NAME.split("/")[-1]
OUTPUT_DIR = project_path("output_dir")
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"  # one subdirectory per run (smoke, EXPERIMENT_DIRNAME)
MAS_OUTPUT_DIR = OUTPUT_DIR / "mas"


def experiment_path(root: Path, percentage: float = CLIP_PERCENTAGE) -> Path:
    if percentage == 100.0:
        return root
    label = str(float(percentage)).removesuffix(".0").replace(".", "p")
    return root / f"clips_{label}pct"


TRAINING_RUN_DIR = experiment_path(TRAINING_OUTPUT_DIR / EXPERIMENT_DIRNAME)
MAS_RUN_DIR = experiment_path(MAS_OUTPUT_DIR / EXPERIMENT_DIRNAME)
DEMO_OUTPUT_DIR = experiment_path(MAS_OUTPUT_DIR / "demo")
REPORT_DIR = experiment_path(OUTPUT_DIR / "report")
LOG_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = OUTPUT_DIR / "cache"
TRAINING_DATA_DIR = CACHE_DIR / "training"  # the merged train/test NPZs, built from RAW_DATA_DIR
TRAINING_MEMMAP_DIR = TRAINING_DATA_DIR / "memmap"  # memory-mapped copies of the NPZ members


assert TARGET_WINDOW_SIZE == len(POSITION_NAMES)
assert SOURCE_GESTURE_SEQUENCE_LENGTH >= TARGET_WINDOW_SIZE
assert PATIENT_FEATURE_DIM == (
    len(PARETIC_SIDE_CATEGORIES)
    + len(GENDER_CATEGORIES)
    + len(RACE_CATEGORIES)
    + 2
)
assert CONDITIONED_VISION_DIM + PATIENT_PREFIX_DIM == QWEN_HIDDEN_SIZE
assert set(TASK_DESCRIPTIONS) == set(TASKS)
assert {mode for _, mode in VARIANTS} == set(CONDITIONING_MODES)
assert sorted(TRAINING_VARIANTS) == sorted(VARIANTS)
assert VARIANTS[0][1] == CONDITIONING_NONE  # the unconditioned baseline of every contrast
assert len(set(SEEDS)) == len(SEEDS)
