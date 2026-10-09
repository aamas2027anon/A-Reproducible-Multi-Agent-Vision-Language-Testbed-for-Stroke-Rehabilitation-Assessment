# Stroke Rehabilitation VLM and Multi-Agent Testbed

This repository contains the supplementary code for training the vision-language movement model (VLM) and running, tuning, and evaluating the multi-agent assessment system (MAS).

## Install

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then create the locked CUDA 12.6 environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked
```

Training needs one CUDA GPU (the paper used a 40 GB A100). The base model `Qwen/Qwen3.5-0.8B-Base` is downloaded from Hugging Face on first use.

## Data

Training, evaluation, and the demo all read the raw StrokeRehab tree (`video_extracted_features/` and `kinematics/`, with each subject's `metadata.xml`). Place or symlink it at `code/data/stroke_rehab` (ignored by git), which every command reads by default:

```bash
ln -s /path/to/stroke_rehab code/data/stroke_rehab
```

Alternatively, set `data_dir` in `code/project_config.toml`, or pass `--raw-dir /path/to/stroke_rehab` to each command.

`code/project_config.toml` is the only place for machine-specific paths (data, training outputs, caches, models, checkpoints); its defaults are relative to `code/`. Outputs and checkpoints record paths relative to `code/`, never absolute ones. `STROKE_REHAB_CONFIG` selects another configuration file.

On the first training command, the raw clips are converted into `stroke_rehab_train_all_tasks.npz` (the 32 training subjects) and `stroke_rehab_test_all_tasks.npz` (the 8 test subjects) under `code/data/training` (`training_data_dir`, or `--data-root`). The MAS builds the same files from the same raw data, so both pipelines see identical windows, labels, and patient fields.

All commands below are run from `code/`:

```bash
cd code
```

## Training

`--stage multiseed` is the run reported in the paper: the three conditioning variants (`none`, `text`, `prefix`) for each of the 10 seeds.

```bash
# All 10 seeds x 3 variants, serially on one GPU
uv run python train/run_pipeline.py multiseed --stage multiseed

# Or one seed per job (index 0-9 into the seed list)
uv run python train/run_pipeline.py seed --stage multiseed --seed-index 0

# VLM variant statistics across seeds (seed-level metrics and paired tests)
uv run python train/run_pipeline.py multiseed-aggregate --stage multiseed
```

Checkpoints and test predictions go to `train/outputs/qwen08_multiseed_counting/seeds/seed_<S>/<variant>/`, where `<variant>` is `qwen_without_patient_conditioning`, `qwen_with_patient_text`, or `qwen_with_patient_prefix`. A seed's checkpoint is `<variant>/checkpoints/<config id>/last.ckpt`. The aggregate tables go to `train/outputs/qwen08_multiseed_counting/seeds/analysis/` and `train/outputs/stat_analysis_report_tables_multiseed/`.

`--stage ceiling` is a separate single-seed (seed 42) ceiling test, run before the multi-seed sweep to check how far the model gets; it is not used for the paper's results:

```bash
uv run python train/run_pipeline.py seed --stage ceiling
```

## Evaluation

The MAS always uses a seed's **text-conditioned** checkpoint (`qwen_with_patient_text`). Evaluation compares five modes on identical cached VLM outputs: `vlm`, `mas`, `mas_no_memory`, `mas_no_planning`, and `mas_no_safety`.

**All seeds (paper results).** For every seed, this tunes planning, the rest penalty, and the UE-FMA calibration on the four validation subjects, evaluates all modes on the test subjects, and then builds the across-seed report:

```bash
uv run python -m multi_agents.multiseed --training-root train/outputs/qwen08_multiseed_counting
```

Outputs go to `multi_agent_outputs/multiseed_text/`: `seed_<S>/tuned_config.json`, `seed_<S>/eval_test/`, and the across-seed tables and plots in `report/` (`report/tables/`, `report/mas_only/tables/`, `report/seed_level_metrics.csv`). Seeds already evaluated are skipped, so an interrupted run resumes.

**One checkpoint.**

```bash
CKPT=train/outputs/qwen08_multiseed_counting/seeds/seed_1337/qwen_with_patient_text/checkpoints/<config id>/last.ckpt

# Tune on the validation subjects (never on test)
uv run python -m multi_agents.tune --split validation --checkpoint $CKPT \
  --output multi_agent_outputs/tuned_config.json

# Evaluate the VLM, full MAS, and ablations on the test subjects
uv run python -m multi_agents.evaluate --split test --checkpoint $CKPT \
  --mas-config multi_agent_outputs/tuned_config.json

# Rebuild the across-seed report from existing seed_<S>/eval_test/ runs
uv run python -m multi_agents.multiseed_report --root multi_agent_outputs/multiseed_text
```

`evaluate` writes `comparison.csv`, `paired_comparison.csv`, per-mode predictions and tables, and figures to `multi_agent_outputs/eval_test_<run id>/` (or `--output-dir`). Use `--modes` to run a subset of modes.

## MAS demo

The demo streams one recorded exercise frame by frame through the agents, as a live sensor would. After the warm-up context, every completed window prints the current and future primitive and the safety verdict, and at the end of the exercise the clinician report is printed. The report, the system details, the per-window recommendations and decisions, and the patient memory are written to the output directory (`--output-dir`). Pass the tuned config so the demo uses the same settings as the evaluation:

```bash
# Stream one exercise (--fast removes real-time pacing; the outputs are the same)
uv run python -m multi_agents.demo --sample-id s04_drinking3_1 --fast \
  --checkpoint $CKPT --mas-config multi_agent_outputs/tuned_config.json

# Replay the patient's earlier exercises first, so the exercise is decided with that history
uv run python -m multi_agents.demo --sample-id s04_drinking3_1 --replay-history \
  --checkpoint $CKPT --mas-config multi_agent_outputs/tuned_config.json

# Predefined scenarios (e.g. sensor dropout, corrupted features)
uv run python -m multi_agents.demo --list-scenarios
uv run python -m multi_agents.demo --scenario sensor_dropout --checkpoint $CKPT
```

The demo keeps patient history in a sqlite store by default, so running it again continues the same patient's record. Use `--memory inmemory` for a fresh, isolated run.

Without `--checkpoint`, every MAS command uses `checkpoint` in `code/project_config.toml`. Every command lists its full options with `--help`.
