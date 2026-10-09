"""Multi-seed MAS run: tune and evaluate every seed's text checkpoint, then the report.

    python -m multi_agents.multiseed --training-root train/outputs/qwen08_multiseed_counting

For each `<training-root>/seeds/seed_<S>/qwen_with_patient_text/checkpoints/*/last.ckpt`
(the MAS always uses the text-conditioned checkpoint of a seed):

  1. `multi_agents.tune` on the validation subjects -> <output>/seed_<S>/tuned_config.json
  2. `multi_agents.evaluate` on the test subjects   -> <output>/seed_<S>/eval_<split>/
     (the training predictions of the same checkpoint are the reference predictions)

and finally `multi_agents.multiseed_report` -> <output>/report/, the paper's MAS tables.
Seeds whose evaluation already finished are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from multi_agents.inference import read_checkpoint_conditioning
from multi_agents.settings import OUTPUT_DIR, RAW_DATA_DIR, REPO_ROOT
from multi_agents.train_bridge import train_config

TEXT_RUN_NAME = next(name for name, mode in train_config.VARIANTS if mode == train_config.CONDITIONING_TEXT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=train_config.default_output_root() / train_config.STAGES[train_config.MULTISEED_STAGE].experiment_group,
        help="Output root of `run_pipeline.py multiseed --stage multiseed` (default: under training_output_dir)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "multiseed_text")
    parser.add_argument("--split", choices=("test", "validation", "none"), default="test")
    parser.add_argument("--modes", default=None, help="Comma list of evaluate.py modes (default: all)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def text_checkpoints(training_root: Path) -> dict[int, Path]:
    """Seed -> last.ckpt of that seed's text-conditioned run."""
    found: dict[int, Path] = {}
    for checkpoint in sorted(training_root.glob(f"seeds/seed_*/{TEXT_RUN_NAME}/checkpoints/*/last.ckpt")):
        seed = int(re.fullmatch(r"seed_(\d+)", checkpoint.parents[3].name).group(1))
        if seed in found:
            raise SystemExit(f"Several text checkpoints for seed {seed}: {found[seed]}, {checkpoint}")
        conditioning, _ = read_checkpoint_conditioning(checkpoint)
        if conditioning != train_config.CONDITIONING_TEXT:
            raise SystemExit(f"{checkpoint} is {conditioning!r}-conditioned; the MAS uses the text checkpoint")
        found[seed] = checkpoint
    if not found:
        raise SystemExit(f"No seeds/seed_<S>/{TEXT_RUN_NAME}/checkpoints/*/last.ckpt under {training_root}")
    return found


def run(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    training_root, output = args.training_root.expanduser().absolute(), args.output.expanduser().absolute()
    checkpoints = text_checkpoints(training_root)
    print(f"{len(checkpoints)} seeds: {sorted(checkpoints)} -> {output}")

    common = ["--raw-dir", args.raw_dir, "--batch-size", args.batch_size]
    if args.base_model:
        common += ["--base-model", args.base_model]
    if args.no_download:
        common.append("--no-download")
    if args.device:
        common += ["--device", args.device]

    for seed, checkpoint in sorted(checkpoints.items()):
        seed_dir = output / f"seed_{seed}"
        evaluation = seed_dir / f"eval_{args.split}"
        if (evaluation / "comparison.csv").is_file():
            print(f"Seed {seed}: already evaluated ({evaluation}); skipping")
            continue
        tuned = seed_dir / "tuned_config.json"
        run("multi_agents.tune", "--split", "validation", "--checkpoint", checkpoint, "--output", tuned, *common)
        extra = []
        # <run>/checkpoints/<config id>/last.ckpt -> <run>/predictions/
        reference = checkpoint.parents[2] / "predictions" / f"{args.split}_combined_window_predictions.csv"
        if reference.is_file():
            extra += ["--reference-predictions", reference]
        if args.modes:
            extra += ["--modes", args.modes]
        run(
            "multi_agents.evaluate", "--split", args.split, "--checkpoint", checkpoint, "--mas-config", tuned,
            "--output-dir", evaluation, *common, *extra,
        )

    run(
        "multi_agents.multiseed_report", "--root", output, "--split", args.split,
        "--training-seed-metrics", training_root / "seeds" / "analysis" / "seed_level_metrics.csv",
    )


if __name__ == "__main__":
    main()
