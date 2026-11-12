"""Multi-seed MAS evaluation: tune and evaluate every seed's text checkpoint, then the report.

    python -m multi_agents.multiseed

For each `<training-root>/seeds/seed_<S>/qwen_with_patient_text/checkpoints/*/best.ckpt`
(the MAS always uses the text-conditioned checkpoint of a seed):

  1. `multi_agents.tune` on the validation subjects -> <output>/seed_<S>/tuned_config.json
  2. `multi_agents.evaluate` on the test subjects   -> <output>/seed_<S>/eval_test/

and finally `multi_agents.report` -> outputs/report/. Seeds whose evaluation already
finished are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from multi_agents.checkpoints import text_checkpoints
from multi_agents.settings import LOG_DIR, RAW_DATA_DIR, REPO_ROOT
from multi_agents.train_bridge import start_log, train_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=train_config.TRAINING_RUN_DIR,
        help="Output root of `run_pipeline.py multiseed`",
    )
    parser.add_argument("--output", type=Path, default=train_config.MAS_RUN_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def run(module: str, *arguments: object) -> None:
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    training_root, output = args.training_root.expanduser().absolute(), args.output.expanduser().absolute()
    start_log(LOG_DIR, "mas_eval")
    checkpoints = text_checkpoints(training_root)
    print(f"{len(checkpoints)} seeds: {sorted(checkpoints)} -> {output}")

    common = ["--raw-dir", args.raw_dir, "--batch-size", args.batch_size]
    if args.base_model:
        common += ["--base-model", args.base_model]
    if args.no_download:
        common.append("--no-download")
    if args.device:
        common += ["--device", args.device]

    for seed, checkpoint in checkpoints.items():
        seed_dir = output / f"seed_{seed}"
        evaluation = seed_dir / "eval_test"
        if (evaluation / "metrics.csv").is_file():
            print(f"Seed {seed}: already evaluated ({evaluation}); skipping")
            continue
        tuned = seed_dir / "tuned_config.json"
        run("multi_agents.tune", "--checkpoint", checkpoint, "--output", tuned, *common)
        run("multi_agents.evaluate", "--checkpoint", checkpoint, "--mas-config", tuned, "--output-dir", evaluation, *common)

    run("multi_agents.report", "--root", output, "--training-root", training_root)


if __name__ == "__main__":
    main()
