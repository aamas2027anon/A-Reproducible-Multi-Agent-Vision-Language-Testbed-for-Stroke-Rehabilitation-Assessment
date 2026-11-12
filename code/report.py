"""Regenerate testbed CSV tables and empirical PNG figures from saved results."""
from __future__ import annotations

import argparse
from pathlib import Path

from multi_agents import settings  # initialize the shared training module path
import config as train_config
from run_log import start_log
from multi_agents.report import export_report as export_mas_report
from train.report import export_report as export_vlm_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path,
                        default=train_config.TRAINING_RUN_DIR)
    parser.add_argument("--mas-root", type=Path, default=train_config.MAS_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=train_config.REPORT_DIR)
    args = parser.parse_args()
    start_log(train_config.LOG_DIR, "report")
    config = train_config.RuntimeConfig(
        data_root=train_config.TRAINING_DATA_DIR,
        cache_root=train_config.TRAINING_MEMMAP_DIR,
        output_root=args.training_root,
    )
    export_vlm_report(config, args.output_dir)
    export_mas_report(args.mas_root, args.output_dir, args.training_root)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
