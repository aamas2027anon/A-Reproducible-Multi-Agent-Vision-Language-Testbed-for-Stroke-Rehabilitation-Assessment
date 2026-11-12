"""Discover the text-conditioned checkpoints used by MAS evaluation and demos."""
from __future__ import annotations

import re
from pathlib import Path

from multi_agents.inference import read_checkpoint_conditioning
from multi_agents.train_bridge import train_config

TEXT_RUN_NAME = next(
    name for name, mode in train_config.VARIANTS if mode == train_config.CONDITIONING_TEXT
)


def text_checkpoints(
    training_root: Path, seeds: tuple[int, ...] = train_config.SEEDS,
) -> dict[int, Path]:
    """Return each selected seed's unique text-conditioned best checkpoint."""
    found: dict[int, Path] = {}
    for checkpoint in sorted(training_root.glob(f"seeds/seed_*/{TEXT_RUN_NAME}/checkpoints/*/best.ckpt")):
        match = re.fullmatch(r"seed_(\d+)", checkpoint.parents[3].name)
        if match is None:
            continue
        seed = int(match.group(1))
        if seed not in seeds:
            continue
        if not checkpoint.is_file():
            continue
        if seed in found:
            raise SystemExit(f"Several text checkpoints for seed {seed}: {found[seed]}, {checkpoint}")
        conditioning, _ = read_checkpoint_conditioning(checkpoint)
        if conditioning != train_config.CONDITIONING_TEXT:
            raise SystemExit(f"{checkpoint} is {conditioning!r}-conditioned; the MAS uses the text checkpoint")
        found[seed] = checkpoint
    if not found:
        raise SystemExit(
            f"No text best.ckpt for selected seeds {list(seeds)} under {training_root}"
        )
    return {seed: found[seed] for seed in seeds if seed in found}
