"""Per-epoch metrics and plots, written to disk while a run trains.

This module only writes files. It never sets `trainer.should_stop`, never selects a
checkpoint, and never compares a metric against a threshold, so the held-out prediction
split reported here has no influence on training. Stopping a run is a manual decision
made by reading these files.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

from artifacts import classification_summary
from config import POSITION_NAMES, RuntimeConfig, TASKS, portable_path
from data import StrokeRehabCollator, StrokeRehabDataset, make_loader
from model import compute_joint_loss, model_batch
from training import build_prediction_frames


EPOCH_METRIC_COLUMNS = (
    "epoch",
    "run_name",
    "conditioning",
    "seed",
    "train_joint_loss",
    "train_primitive_loss",
    "train_ue_fma_normalized_mse",
    "train_accuracy",
    "val_joint_loss",
    "val_primitive_loss",
    "val_ue_fma_normalized_mse",
    "val_accuracy",
    "test_joint_loss",
    "test_primitive_loss",
    "test_ue_fma_normalized_mse",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_reach_f1",
    "test_reposition_f1",
)

_LOSS_PANELS = (
    ("joint_loss", "Joint loss"),
    ("primitive_loss", "Primitive cross-entropy"),
    ("ue_fma_normalized_mse", "UE-FMA normalized MSE"),
)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a sibling temp file so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        # The temp name has no image extension, so the format must be explicit.
        figure.savefig(temporary, format="png", dpi=200, bbox_inches="tight")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def read_lightning_history(history_path: Path) -> pd.DataFrame:
    """Collapse a Lightning CSVLogger metrics file to one row per epoch."""
    if not history_path.is_file():
        return pd.DataFrame(columns=["epoch"])
    history = pd.read_csv(history_path)
    if "epoch" not in history:
        return pd.DataFrame(columns=["epoch"])
    history["epoch"] = pd.to_numeric(history["epoch"], errors="coerce")
    history = history.dropna(subset=["epoch"])
    numeric = [
        column
        for column in history.columns
        if column != "epoch" and pd.api.types.is_numeric_dtype(history[column])
    ]
    if not numeric:
        return pd.DataFrame(columns=["epoch"])
    collapsed = history.groupby("epoch", as_index=False)[numeric].mean(numeric_only=True)
    collapsed["epoch"] = collapsed["epoch"].astype(int)
    return collapsed.sort_values("epoch").reset_index(drop=True)


def task_metric_rows(
    combined: pd.DataFrame, epoch: int, run_name: str, conditioning: str, seed: int
) -> list[dict[str, Any]]:
    """Per-task, per-horizon classification metrics for one epoch."""
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_frame = combined.loc[combined["task_name"] == task]
        if task_frame.empty:
            continue
        for horizon, position in enumerate(POSITION_NAMES):
            selected = task_frame.loc[task_frame["horizon"] == horizon]
            if selected.empty:
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "run_name": run_name,
                    "conditioning": conditioning,
                    "seed": seed,
                    "task_name": task,
                    "prediction_position": position,
                    **classification_summary(selected),
                }
            )
    return rows


def plot_train_vs_validation(history: pd.DataFrame, plot_dir: Path) -> None:
    if history.empty:
        return
    figure, axes = plt.subplots(1, len(_LOSS_PANELS), figsize=(16, 4.6), squeeze=False)
    for axis, (suffix, label) in zip(axes[0], _LOSS_PANELS):
        for split, color in (("train", "#2f6b8a"), ("val", "#c05a2b")):
            column = f"{split}_{suffix}"
            if column not in history:
                continue
            series = history[["epoch", column]].dropna()
            axis.plot(
                series["epoch"] + 1,
                series[column],
                marker="o",
                color=color,
                label=split,
            )
        axis.set(xlabel="Epoch", ylabel=label, title=f"Train vs validation: {label}")
        axis.grid(alpha=0.25)
        axis.legend()
    atomic_save_figure(figure, plot_dir / "train_vs_val_loss.png")

    figure, axis = plt.subplots(figsize=(7.5, 5))
    for split, color in (("train", "#2f6b8a"), ("val", "#c05a2b")):
        column = f"{split}_accuracy"
        if column not in history:
            continue
        series = history[["epoch", column]].dropna()
        axis.plot(series["epoch"] + 1, series[column], marker="o", color=color, label=split)
    axis.set(
        xlabel="Epoch",
        ylabel="Primitive accuracy",
        title="Train vs validation accuracy",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    atomic_save_figure(figure, plot_dir / "train_vs_val_accuracy.png")


def plot_test_by_epoch(epochs: pd.DataFrame, plot_dir: Path) -> None:
    if epochs.empty:
        return
    figure, axes = plt.subplots(1, len(_LOSS_PANELS), figsize=(16, 4.6), squeeze=False)
    for axis, (suffix, label) in zip(axes[0], _LOSS_PANELS):
        column = f"test_{suffix}"
        if column not in epochs:
            continue
        series = epochs[["epoch", column]].dropna()
        axis.plot(series["epoch"] + 1, series[column], marker="o", color="#2e7d32")
        axis.set(xlabel="Epoch", ylabel=label, title=f"Test: {label}")
        axis.grid(alpha=0.25)
    atomic_save_figure(figure, plot_dir / "test_loss_by_epoch.png")

    figure, axis = plt.subplots(figsize=(7.5, 5))
    for column, label, color in (
        ("test_accuracy", "Accuracy", "#2e7d32"),
        ("test_macro_f1", "Macro F1", "#7b4fa3"),
        ("test_weighted_f1", "Weighted F1", "#c05a2b"),
    ):
        if column not in epochs:
            continue
        series = epochs[["epoch", column]].dropna()
        axis.plot(series["epoch"] + 1, series[column], marker="o", color=color, label=label)
    axis.set(xlabel="Epoch", ylabel="Score", title="Test accuracy and F1 by epoch")
    axis.grid(alpha=0.25)
    axis.legend()
    atomic_save_figure(figure, plot_dir / "test_accuracy_by_epoch.png")


def plot_task_accuracy(task_metrics: pd.DataFrame, plot_dir: Path) -> None:
    if task_metrics.empty:
        return
    current = task_metrics.loc[task_metrics["prediction_position"] == POSITION_NAMES[0]]
    if current.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for task, task_frame in current.groupby("task_name", sort=True):
        series = task_frame.sort_values("epoch")
        axis.plot(series["epoch"] + 1, series["accuracy"], marker="o", label=task)
    axis.set(
        xlabel="Epoch",
        ylabel="Accuracy (current position)",
        title="Test accuracy by task",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    atomic_save_figure(figure, plot_dir / "task_accuracy_by_epoch.png")


class EpochMetricsCallback(Callback):
    """Write metrics and plots for the held-out split at the end of every epoch."""

    def __init__(
        self,
        dataset: StrokeRehabDataset,
        collator: StrokeRehabCollator,
        config: RuntimeConfig,
        run_name: str,
        conditioning: str,
        seed: int,
        epoch_dir: Path,
        split: str = "test",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.config = config
        self.run_name = run_name
        self.conditioning = conditioning
        self.seed = int(seed)
        self.epoch_dir = Path(epoch_dir)
        self.split = split
        self.metrics_path = self.epoch_dir / "epoch_metrics.csv"
        self.task_metrics_path = self.epoch_dir / "epoch_task_metrics.csv"
        self.manifest_path = self.epoch_dir / "epoch_manifest.json"
        self.plot_dir = self.epoch_dir / "plots"
        self._loader: Any | None = None
        # Reload what a previous, interrupted run already wrote so a resumed job keeps
        # its earlier epochs instead of restarting the table at the resume point.
        self._rows = self._existing_rows(self.metrics_path)
        self._task_rows = self._existing_rows(self.task_metrics_path)

    @staticmethod
    def _existing_rows(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            return pd.read_csv(path).to_dict("records")
        except Exception:
            print(f"Could not reuse {path}; starting a fresh table")
            return []

    # -- Lightning hook ---------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        try:
            self._write_epoch(trainer, pl_module, epoch)
        except Exception:
            # A plotting or IO fault must never take down a multi-epoch training run.
            print(f"Per-epoch metrics failed for epoch {epoch}; recording NaNs:")
            traceback.print_exc()
            self._rows.append(self._nan_row(epoch))
            self._persist()

    # -- Internals --------------------------------------------------------------

    def _nan_row(self, epoch: int) -> dict[str, Any]:
        row = {name: np.nan for name in EPOCH_METRIC_COLUMNS}
        row.update(
            {
                "epoch": epoch,
                "run_name": self.run_name,
                "conditioning": self.conditioning,
                "seed": self.seed,
            }
        )
        return row

    def _prediction_loader(self) -> Any:
        """Build the evaluation loader once; workers then persist across epochs."""
        if self._loader is None:
            self._loader = make_loader(
                self.dataset, self.collator, self.config, shuffle=False, seed=self.seed
            )
        return self._loader

    def release(self) -> None:
        """Drop the evaluation loader and its persistent workers.

        Called at the variant boundary: the workers hold page-locked host buffers, and
        without this they stay attached while the next variant builds its own model.
        """
        loader, self._loader = self._loader, None
        if loader is None:
            return
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None) if iterator else None
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                print("Could not shut down per-epoch loader workers cleanly")

    @torch.no_grad()
    def _predict(self, pl_module: L.LightningModule) -> tuple[pd.DataFrame, dict[str, float]]:
        """Run the held-out split through the in-memory module."""
        loader = self._prediction_loader()
        was_training = pl_module.training
        pl_module.eval()
        batches: list[Mapping[str, Any]] = []
        loss_totals = {"total": 0.0, "primitive": 0.0, "ue_fma_normalized": 0.0}
        sample_count = 0
        # Match the trainer's mixed precision so this pass costs about what a
        # validation pass costs.
        autocast_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        use_autocast = pl_module.device.type == "cuda"
        try:
            for batch in loader:
                device_batch = {
                    key: value.to(pl_module.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=use_autocast
                ):
                    logits, predicted_fma = pl_module(**model_batch(device_batch))
                    losses = compute_joint_loss(
                        logits,
                        predicted_fma,
                        device_batch["gesture_labels"],
                        device_batch["ue_fma_score"],
                    )
                size = int(device_batch["gesture_labels"].shape[0])
                sample_count += size
                loss_totals["total"] += float(losses.total) * size
                loss_totals["primitive"] += float(losses.primitive) * size
                loss_totals["ue_fma_normalized"] += float(losses.ue_fma_normalized) * size
                probabilities = torch.softmax(logits.float(), dim=-1)
                predicted_ids = probabilities.argmax(dim=-1)
                batches.append(
                    {
                        "labels": device_batch["gesture_labels"].detach().cpu(),
                        "predicted_ids": predicted_ids.detach().cpu(),
                        "predicted_probabilities": probabilities.gather(
                            -1, predicted_ids.unsqueeze(-1)
                        )
                        .squeeze(-1)
                        .detach()
                        .cpu(),
                        "actual_fma": device_batch["ue_fma_score"].detach().float().cpu(),
                        "predicted_fma": predicted_fma.detach().float().cpu(),
                        "subject_id": list(batch["subject_id"]),
                        "clip_id": list(batch["clip_id"]),
                        "task_name": list(batch["task_name"]),
                        "target_frame_index": list(batch["target_frame_index"]),
                    }
                )
        finally:
            if was_training:
                pl_module.train()
        divisor = max(sample_count, 1)
        losses_mean = {key: value / divisor for key, value in loss_totals.items()}
        _, _, combined = build_prediction_frames(
            batches, self.run_name, -1, self.split
        )
        return combined, losses_mean

    def _write_epoch(
        self, trainer: L.Trainer, pl_module: L.LightningModule, epoch: int
    ) -> None:
        combined, losses = self._predict(pl_module)
        summary = classification_summary(combined)
        # Read train/val from callback_metrics rather than the CSV: the logger has not
        # necessarily flushed this epoch's row yet, and these values are authoritative.
        callback_metrics = trainer.callback_metrics
        history = read_lightning_history(Path(trainer.logger.log_dir) / "metrics.csv")
        logged = history.loc[history["epoch"] == epoch] if not history.empty else history

        def logged_value(column: str) -> float:
            value = callback_metrics.get(column)
            if value is not None:
                return float(value)
            if logged.empty or column not in logged:
                return float("nan")
            return float(logged[column].iloc[0])

        row: dict[str, Any] = {
            "epoch": epoch,
            "run_name": self.run_name,
            "conditioning": self.conditioning,
            "seed": self.seed,
            "test_joint_loss": losses["total"],
            "test_primitive_loss": losses["primitive"],
            "test_ue_fma_normalized_mse": losses["ue_fma_normalized"],
            "test_accuracy": summary["accuracy"],
            "test_macro_f1": summary["macro_f1"],
            "test_weighted_f1": summary["weighted_f1"],
            "test_reach_f1": summary["reach_f1"],
            "test_reposition_f1": summary["reposition_f1"],
        }
        for split in ("train", "val"):
            for suffix in ("joint_loss", "primitive_loss", "ue_fma_normalized_mse", "accuracy"):
                column = f"{split}_{suffix}"
                row[column] = logged_value(column)
        self._rows.append(row)
        self._task_rows.extend(
            task_metric_rows(combined, epoch, self.run_name, self.conditioning, self.seed)
        )
        self._persist()

    def _persist(self) -> None:
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        epochs = pd.DataFrame(self._rows)
        if not epochs.empty:
            ordered = [name for name in EPOCH_METRIC_COLUMNS if name in epochs]
            extra = [name for name in epochs.columns if name not in ordered]
            # A resumed run can re-evaluate an epoch it already wrote; the newest row
            # for an epoch wins so the table stays one row per epoch.
            epochs = (
                epochs[[*ordered, *extra]]
                .drop_duplicates(subset="epoch", keep="last")
                .sort_values("epoch")
                .reset_index(drop=True)
            )
            atomic_write_text(self.metrics_path, epochs.to_csv(index=False))
        task_metrics = pd.DataFrame(self._task_rows)
        if not task_metrics.empty:
            task_metrics = task_metrics.drop_duplicates(
                subset=["epoch", "task_name", "prediction_position"], keep="last"
            ).reset_index(drop=True)
            atomic_write_text(self.task_metrics_path, task_metrics.to_csv(index=False))
        # Everything is plotted from the accumulated table, so the figures never lag
        # behind the logger's flush schedule.
        plot_train_vs_validation(epochs, self.plot_dir)
        plot_test_by_epoch(epochs, self.plot_dir)
        plot_task_accuracy(task_metrics, self.plot_dir)
        atomic_write_text(
            self.manifest_path,
            json.dumps(
                {
                    "run_name": self.run_name,
                    "conditioning": self.conditioning,
                    "seed": self.seed,
                    "split": self.split,
                    "epochs_written": len(epochs),
                    "last_epoch": int(epochs["epoch"].max()) if not epochs.empty else None,
                    "subjects": sorted(self.dataset.subjects),
                    # These numbers are reported only; nothing here selects a
                    # checkpoint or stops training.
                    "test_used_for_selection": False,
                    "epoch_metrics": portable_path(self.metrics_path),
                    "epoch_task_metrics": portable_path(self.task_metrics_path),
                    "plots": sorted(portable_path(path) for path in self.plot_dir.glob("*.png")),
                },
                indent=2,
            ),
        )


def epoch_metrics_callback_factory(
    config: RuntimeConfig,
    run_name: str,
    conditioning: str,
    epoch_dir: Path,
    split: str = "test",
) -> Any:
    """Build the factory `run_experiment` calls once its datasets exist.

    The factory records the callbacks it made so the caller can release them at the
    variant boundary with `release_epoch_metrics_callbacks`.
    """
    created: list[EpochMetricsCallback] = []

    def factory(
        dataset: StrokeRehabDataset, collator: StrokeRehabCollator, seed: int
    ) -> Sequence[Callback]:
        callback = EpochMetricsCallback(
            dataset=dataset,
            collator=collator,
            config=config,
            run_name=run_name,
            conditioning=conditioning,
            seed=seed,
            epoch_dir=epoch_dir,
            split=split,
        )
        created.append(callback)
        return (callback,)

    factory.created = created
    return factory


def release_epoch_metrics_callbacks(factory: Any) -> None:
    """Release every callback a factory produced. Safe to call more than once."""
    for callback in getattr(factory, "created", ()):
        callback.release()
    getattr(factory, "created", []).clear()
