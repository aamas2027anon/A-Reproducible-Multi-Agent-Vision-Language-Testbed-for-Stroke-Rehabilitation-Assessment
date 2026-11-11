"""The Hugging Face base model, kept in one local copy under models_dir."""
from __future__ import annotations

from pathlib import Path

from config import BASE_MODEL_DIR, MODEL_NAME

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


def model_source(model_name: str) -> str:
    """Where to load `model_name` from: the base model's local copy (downloaded on first
    use), or `model_name` itself when it is already a path."""
    return str(ensure_model(MODEL_NAME, BASE_MODEL_DIR)) if model_name == MODEL_NAME else model_name
