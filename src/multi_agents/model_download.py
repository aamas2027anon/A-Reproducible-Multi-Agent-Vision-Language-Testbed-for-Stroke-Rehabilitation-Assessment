"""Check that the local base model is the Base model the checkpoints were trained on."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

def warn_if_not_base_model(local_dir: Path) -> None:
    """The LoRA checkpoint needs the Base weights; a post-trained model degrades it.

    Hugging Face model cards of derived models list `base_model:` in their front
    matter, which the Base model's own card does not.
    """
    readme = Path(local_dir) / "README.md"
    if "base" in Path(local_dir).name.lower() or not readme.is_file():
        return
    front_matter = readme.read_text(encoding="utf-8", errors="ignore").split("---")
    if len(front_matter) > 2 and re.search(r"^base_model:", front_matter[1], re.MULTILINE):
        warnings.warn(
            f"{local_dir} looks like a post-trained model, but the trained checkpoint "
            "was fit on Qwen/Qwen3.5-0.8B-Base. Predictions will not match training.",
            stacklevel=2,
        )
