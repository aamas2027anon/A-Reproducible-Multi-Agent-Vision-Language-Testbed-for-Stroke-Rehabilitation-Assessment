# Stroke Rehabilitation VLM and Multi-Agent Testbed

## Setup

An NVIDIA A100 GPU is recommended.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available, then restart your shell.

From the repository root, create the Python 3.12 virtual environment (`.venv/`) and install the project dependencies from `uv.lock`:

```bash
uv venv --python 3.12
uv sync --locked
```

Install the additional attention kernels into the same environment:

```bash
uv pip install "causal-conv1d==1.7.0+cu.12.6.torch.2.13" --extra-index-url https://wheels.astral.sh/simple/cu126 --no-deps
uv pip install "flash-attn==2.8.3.post1+cu.12.6.torch.2.13" --extra-index-url https://wheels.astral.sh/simple/cu126 --no-deps
uv pip install "fla-core==0.5.0" "flash-linear-attention==0.5.0" --no-deps
```

These kernels are installed separately from the lock file; reinstall them after `uv sync`. Commands using `uv run` automatically use `.venv/`; manual activation is optional (`source .venv/bin/activate` from the repository root). The base model downloads automatically on first use.

## Release checkpoint and demo environment

[release_checkpoint/text_seed7331.ckpt](release_checkpoint/text_seed7331.ckpt) is the release checkpoint: the +text variant with the highest saved test weighted F1 across the old ten-seed runs (seed 7331, F1 **0.64362**, accuracy **64.57%**). It is a copy of the retained final-epoch `last.ckpt`, not a validation-best epoch checkpoint. Matching MAS settings and selection details are included beside it. These are previously measured scores, not new measurements.

For demo/UI use without the training environment, create a separate `.venv-demo` from the repository root. The demo requirements use platform-native PyTorch without a CUDA suffix or CUDA-only index.

macOS/Linux:

```bash
uv venv --python 3.12 .venv-demo
uv pip install --python .venv-demo/bin/python -r requirements-demo.txt
cd code
../.venv-demo/bin/python -m demo_ui
```

Windows (PowerShell):

```powershell
uv venv --python 3.12 .venv-demo
uv pip install --python .venv-demo\Scripts\python.exe -r requirements-demo.txt
cd code
..\.venv-demo\Scripts\python.exe -m demo_ui
```

The UI defaults to the configured release checkpoint and its packaged tuned settings. For the CLI:

```bash
../.venv-demo/bin/python -m multi_agents.demo --sample-id s04_drinking3_1 --fast \
  --checkpoint ../release_checkpoint/text_seed7331.ckpt \
  --mas-config ../release_checkpoint/tuned_config.json --device cpu
```

On Windows, use the `Scripts\python.exe` interpreter above for CLI commands too. The UI defaults to automatic device selection; `cpu` does not require a GPU. For NVIDIA acceleration on Linux/Windows, choose a compatible build using the [PyTorch installer](https://pytorch.org/get-started/locally/) and select `cuda` in the UI. Use a platform supported by PyTorch 2.12; these instructions do not guarantee wheels for older Intel Macs.

This leaves `.venv`, `pyproject.toml` and `uv.lock` unchanged. The demo environment excludes Lightning and optional CUDA attention kernels; PyTorch fallbacks may be slower. Set `paths.data_dir` before use; the separate Base model weights download on first use. Use the explicit `.venv-demo` interpreter above to avoid syncing the full training project.

## Configuration

Edit [code/config.yaml](code/config.yaml):

| Setting | Purpose |
|---|---|
| `paths.data_dir` | StrokeRehab directory containing `video_extracted_features/` and `kinematics/` |
| `paths.models_dir`, `paths.output_dir` | Downloaded models and generated outputs |
| `paths.checkpoint` | Optional checkpoint for single-checkpoint commands; overridden by `--checkpoint` |
| `training.num_seeds`, `training.seeds` | Use the first N seeds; generate additional unique seeds deterministically if needed |
| `training.clip_percentage` | Percentage of complete clips per fitting subject (training set), greater than 0 and at most 100; validation and test splits remain complete |
| `training.epochs`, `batch_size`, `gradient_accumulation`, `learning_rate`, `num_workers` | Training hyperparameters; these keys are under `training` |
| `lora`, `mas.agents`, `mas.tuning` | Adaptation settings, agent thresholds and validation tuning grids |

For a short run, set `num_seeds: 1`, `clip_percentage: 3.0` and `epochs: 2` under `training`. Relative YAML paths resolve against the directory containing the YAML file; select another file with `STROKE_REHAB_CONFIG=/absolute/path/config.yaml`. CLI options override their corresponding defaults.

## Run

Run commands from `code/`:

```bash
cd code

# Optional setup check: a few batches per conditioning variant.
uv run python train/run_pipeline.py smoke

# Train the configured seeds and generate VLM reports.
uv run python train/run_pipeline.py multiseed

# Or train one configured seed by its zero-based index and override the maximum epochs.
uv run python train/run_pipeline.py multiseed --seed-index 0 --epochs 2

# Tune and evaluate available text-conditioned checkpoints for the configured seeds, then report.
uv run python -m multi_agents.multiseed --device cuda

# Regenerate reports from saved training and MAS outputs without inference.
uv run python -m report
```

> **Demo:** run the demo (browser UI or single-clip CLI) from the separate demo environment, not with `uv run`. `uv run` syncs the CUDA-only training environment. Set up `.venv-demo` once as described in [Release checkpoint and demo environment](#release-checkpoint-and-demo-environment), then activate it and launch the UI from `code/`:
>
> ```bash
> source .venv-demo/bin/activate      # Windows: .venv-demo\Scripts\Activate.ps1
> cd code
> python -m demo_ui                   # open http://127.0.0.1:7860
> ```
>
> The UI uses the release checkpoint (`paths.checkpoint`) and its packaged `tuned_config.json` by default. For the command-line single-clip demo, run `python -m multi_agents.demo --sample-id s04_drinking3_1 --fast --checkpoint ../release_checkpoint/text_seed7331.ckpt --mas-config ../release_checkpoint/tuned_config.json --device cpu` from the same activated environment.

Training runs `none`, `text` and `prefix` conditioning together. MAS evaluation runs `vlm`, `mas`, `mas_no_memory`, `mas_no_planning` and `mas_no_safety` together. `best.ckpt` is used for evaluation; `last.ckpt` is used for resuming.

For a specific checkpoint, set `CKPT` to its `best.ckpt` path and `TUNED` to the corresponding tuning output path:

```bash
uv run python -m multi_agents.tune --checkpoint "$CKPT" --output "$TUNED" --device cuda
uv run python -m multi_agents.evaluate --checkpoint "$CKPT" --mas-config "$TUNED" --device cuda
uv run python -m multi_agents.demo --sample-id s04_drinking3_1 --replay-history --fast \
  --checkpoint "$CKPT" --mas-config "$TUNED" --device cuda
```

| Command | Useful options |
|---|---|
| Training | `--seed-index`, `--epochs`, `--batch-size`, `--num-workers`, `--output-root` |
| Batch MAS evaluation | `--training-root`, `--output`, `--device`, `--batch-size` |
| Single-clip demo | `--sample-id`, `--history-clips`, `--replay-history`, `--memory`, `--output-dir` |
| Demo UI | `--host`, `--port`; demo options are controls in the browser |
| Report regeneration | `--training-root`, `--mas-root`, `--output-dir` |

Every command supports `--help`.

Outputs default to `outputs/{training,mas,report,logs,cache}/` at the repository root. Partial-data runs use `clips_<percentage>pct` subdirectories. One seed produces descriptive CSVs; multiple seeds add across-seed reports. Completed training seeds and MAS evaluations are reused; choose a different `paths.output_dir` for an independent experiment. Each demo launch creates a separate timestamped directory.

## License

The testbed code is licensed under the [MIT License](LICENSE). Datasets and pretrained model weights retain their own licenses.
