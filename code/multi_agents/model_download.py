"""Fetch the Hugging Face models into `models/` on first use."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

REQUIRED_FILES = ("config.json", "tokenizer_config.json")


def model_is_complete(local_dir: Path) -> bool:
    local_dir = Path(local_dir)
    return all((local_dir / name).is_file() for name in REQUIRED_FILES) and any(
        local_dir.glob("*.safetensors")
    )


def ensure_model(repo_id: str, local_dir: str | Path, allow_download: bool = True) -> Path:
    """Return a complete local copy of `repo_id`, downloading it if needed."""
    local_dir = Path(local_dir).expanduser().resolve()
    if model_is_complete(local_dir):
        return local_dir
    if not allow_download:
        raise FileNotFoundError(
            f"{local_dir} does not contain a complete copy of {repo_id}. "
            "Run without --no-download, or download it with "
            f"`hf download {repo_id} --local-dir {local_dir}`."
        )
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} to {local_dir} (first run only) ...", flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    if not model_is_complete(local_dir):
        raise RuntimeError(f"Download of {repo_id} finished but {local_dir} is incomplete")
    return local_dir


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
