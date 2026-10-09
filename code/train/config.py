from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"

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
MAX_TEXT_LENGTH = 256
PREDICTION_PROMPT = "Predict five movement primitives and one UE-FMA score."

NUM_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 4
NUM_WORKERS = 12
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
# Constant learning rate: StepLR is kept so gamma can be dialled back later without a
# code change, but at 1.0 it is a no-op. Decaying at 0.6/epoch put the LR at 6e-8 by
# epoch 10, which made a long run measure the schedule expiring rather than the model
# saturating. Expect validation loss to eventually turn upward instead of flattening;
# every epoch is checkpointed, so the best one stays recoverable.
SCHEDULER_GAMMA = 1.0
GRAD_CLIP_NORM = 1.0
UEFMA_LOSS_WEIGHT = 0.1
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 1
EARLY_STOPPING_MIN_DELTA = 0.01
AUTO_RESUME = True
USE_GRADIENT_CHECKPOINTING = False
DETERMINISTIC_TRAINING = False

LORA_R = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
)

CV_FOLD_OPTIONS = (4, 8)
N_FOLDS = 8
EXPECTED_TRAIN_SUBJECTS = 32
EXPECTED_TEST_SUBJECTS = 8
FULL_VALIDATION_SUBJECTS = frozenset({"s03", "s19", "s27", "s32"})
SEED = 9575
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
UEFMA_TARGET_R = 0.65

# Multi-seed replication of the full train/test stage. The list is written out
# explicitly rather than drawn from a generator so that any individual seed can be
# re-run on its own and land on exactly the same value. A malformed list (repeats,
# too few entries) is repaired at startup by multiseed.resolve_seeds instead of
# aborting a queued job; see that function for the repair rules.
N_SEEDS = 10
SEEDS = (9575, 1337, 2718, 3141, 4242, 5150, 6023, 7331, 8191, 9001)

# The ceiling test runs the three variants once, to answer how far the model gets
# before committing GPU-weeks to the full sweep.
CEILING_SEED = 42

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
EVENT_IOU_THRESHOLD = 0.5
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


@dataclass(frozen=True)
class StageSpec:
    """One independently submittable run of the pipeline.

    Stages have separate output roots, report directories, and — because the experiment
    group is hashed into the checkpoint id — separate checkpoint namespaces.
    """

    name: str
    experiment_group: str
    report_dirname: str
    epochs: int
    seeds: tuple[int, ...]
    n_folds: int

    @property
    def is_seed_driven(self) -> bool:
        return bool(self.seeds)


CEILING_STAGE = "ceiling"
MULTISEED_STAGE = "multiseed"
CROSSVAL_STAGE = "crossval"

STAGES = {
    CEILING_STAGE: StageSpec(
        name=CEILING_STAGE,
        experiment_group="qwen08_ceiling_counting",
        report_dirname="stat_analysis_report_tables_ceiling",
        epochs=NUM_EPOCHS,
        seeds=(CEILING_SEED,),
        n_folds=N_FOLDS,
    ),
    MULTISEED_STAGE: StageSpec(
        name=MULTISEED_STAGE,
        experiment_group="qwen08_multiseed_counting",
        report_dirname="stat_analysis_report_tables_multiseed",
        epochs=NUM_EPOCHS,
        seeds=SEEDS,
        n_folds=N_FOLDS,
    ),
    CROSSVAL_STAGE: StageSpec(
        name=CROSSVAL_STAGE,
        experiment_group="qwen08_crossval_counting",
        report_dirname="stat_analysis_report_tables_crossval",
        epochs=NUM_EPOCHS,
        seeds=(),
        n_folds=N_FOLDS,
    ),
}
STAGE_NAMES = tuple(STAGES)
DEFAULT_STAGE = CEILING_STAGE
MULTISEED_REPORT_DIRNAME = STAGES[MULTISEED_STAGE].report_dirname


def stage_spec(name: str) -> StageSpec:
    if name not in STAGES:
        raise ValueError(f"Unknown stage {name!r}; expected one of {STAGE_NAMES}")
    return STAGES[name]


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path
    output_root: Path
    report_root: Path
    cache_root: Path
    model_name: str = MODEL_NAME
    seed: int = SEED
    batch_size: int = PER_DEVICE_BATCH_SIZE
    num_workers: int = NUM_WORKERS
    epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    resume: bool = AUTO_RESUME
    n_folds: int = N_FOLDS
    stage: str = DEFAULT_STAGE

    @property
    def stage_spec(self) -> StageSpec:
        return stage_spec(self.stage)

    @property
    def experiment_group(self) -> str:
        """Hashed into the checkpoint id, so it is what isolates the stages."""
        return self.stage_spec.experiment_group

    @property
    def train_data_path(self) -> Path:
        return self.data_root / "stroke_rehab_train_all_tasks.npz"

    @property
    def test_data_path(self) -> Path:
        return self.data_root / "stroke_rehab_test_all_tasks.npz"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("data_root", "output_root", "report_root", "cache_root"):
            result[key] = portable_path(result[key])
        if os.path.isabs(self.model_name):
            result["model_name"] = portable_path(self.model_name)
        result["experiment_group"] = self.experiment_group
        return result


def default_cval_root() -> Path:
    return Path(__file__).resolve().parent


def _project_paths() -> tuple[Path, dict[str, Any]]:
    """`[paths]` of project_config.toml (or of STROKE_REHAB_CONFIG) and the root they resolve against."""
    config_file = Path(
        os.environ.get("STROKE_REHAB_CONFIG", default_cval_root().parent / "project_config.toml")
    ).expanduser().absolute()
    paths = tomllib.loads(config_file.read_text(encoding="utf-8")).get("paths", {}) if config_file.is_file() else {}
    root = Path(str(paths.get("project_root", "."))).expanduser()
    return Path(os.path.normpath(config_file.parent / root)), paths


PROJECT_ROOT, _PROJECT_PATHS = _project_paths()


def project_path(key: str, default: str) -> Path:
    """A path from project_config.toml; relative values resolve against the project root."""
    value = Path(str(_PROJECT_PATHS.get(key, default))).expanduser()
    return Path(os.path.normpath(PROJECT_ROOT / value))


def portable_path(path: Path | str) -> str:
    """How a path is recorded in manifests and checkpoints: relative to the project root.

    A location outside the project (an absolute path set in project_config.toml) is
    recorded by its name only, so no output carries a machine-specific path.
    """
    path = Path(path).expanduser()
    for candidate, root in (
        (Path(os.path.normpath(path.absolute())), PROJECT_ROOT),
        (path.resolve(), PROJECT_ROOT.resolve()),
    ):
        if candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
    return f"<external>/{path.name}"


def default_data_root() -> Path:
    """The merged train/test NPZs, built from the raw data if missing."""
    return project_path("training_data_dir", "data/training")


def default_raw_dir() -> Path:
    """The raw StrokeRehab tree, shared with the multi-agent system."""
    return project_path("data_dir", "data/stroke_rehab")


def default_output_root() -> Path:
    """Checkpoints, predictions and reports; each stage writes to its own subdirectory."""
    return project_path("training_output_dir", "train/outputs")


def default_cache_root() -> Path:
    """Memory-mapped copies of the NPZ members."""
    return project_path("training_cache_dir", "train/cache")


assert N_FOLDS in CV_FOLD_OPTIONS
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
# Stages must not share an output root or a checkpoint namespace.
assert len({spec.experiment_group for spec in STAGES.values()}) == len(STAGES)
assert len({spec.report_dirname for spec in STAGES.values()}) == len(STAGES)
assert DEFAULT_STAGE in STAGES
assert STAGES[CEILING_STAGE].seeds == (CEILING_SEED,)
