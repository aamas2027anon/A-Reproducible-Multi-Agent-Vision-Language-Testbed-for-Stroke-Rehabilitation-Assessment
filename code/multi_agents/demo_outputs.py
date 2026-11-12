"""Decision traces and empirical plots for real exercise replay."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from multi_agents import metrics
from multi_agents.settings import PRIMITIVES

COLORS = ("#4c6ef5", "#f08c00", "#adb5bd", "#2f9e44", "#ae3ec9")


def write_records(records: list[dict], path: Path) -> None:
    rows = [{key: json.dumps(value) if isinstance(value, (dict, list)) else value
             for key, value in record.items()} for record in records]
    pd.DataFrame(rows).to_csv(path, index=False)


def decision_timeline(records: pd.DataFrame, primitives: pd.DataFrame, path: Path) -> None:
    frame = primitives.sort_values("absolute_frame_index")
    fig, axes = plt.subplots(5, 1, figsize=(11, 4.5), sharex=True, constrained_layout=True)
    for axis, column, label in zip(axes[:4],
                                  ("actual_label_id", "model_label_id", "plan_label_id", "predicted_label_id"),
                                  ("Actual", "Perception", "Planning", "Final")):
        axis.imshow(frame[column].to_numpy()[None, :], aspect="auto", interpolation="nearest",
                    cmap=ListedColormap(COLORS), vmin=-0.5, vmax=4.5,
                    extent=(frame.absolute_frame_index.min(), frame.absolute_frame_index.max() + 1, 0, 1))
        axis.set_yticks([])
        axis.set_ylabel(label, rotation=0, ha="right", va="center")
    statuses = ("approve", "revise", "escalate", "block")
    axes[4].step(records.target_frame_index, records.status.map(statuses.index), where="post", color="#343a40")
    axes[4].set_yticks(range(4), statuses)
    axes[4].set_ylim(-0.3, 3.3)
    axes[4].set_ylabel("Status")
    axes[4].set_xlabel("Frame")
    for record in records.loc[~records.released].itertuples():
        axes[3].axvspan(record.target_frame_index, record.target_frame_index + 5,
                       facecolor="white", alpha=0.65, hatch="///", edgecolor="#555555")
    axes[0].set_title(f"{records.clip_id.iloc[0]} — hatched final predictions were withheld")
    axes[0].legend(handles=[Patch(color=color, label=name) for color, name in zip(COLORS, PRIMITIVES)],
                   loc="lower center", bbox_to_anchor=(0.5, 1.55), ncol=5, fontsize=8)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def patient_trajectory(records: list[dict], config, truth: float, path: Path, preview: bool) -> None:
    frame = pd.DataFrame(records)
    x = np.arange(1, len(frame) + 1)
    estimate = pd.to_numeric(frame.ue_fma_estimate_after).to_numpy(dtype=float)
    interval = pd.to_numeric(frame.ue_fma_ci95).to_numpy(dtype=float)
    exercise = config.calibrate_uefma(pd.to_numeric(frame.ue_fma_exercise).to_numpy(dtype=float))
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(x, exercise, ".-", color="#adb5bd", label="Exercise mean")
    axes[0].plot(x, estimate, ".-", color="#4c6ef5", label="Pooled estimate")
    axes[0].fill_between(x, estimate - interval, estimate + interval, alpha=0.15, color="#4c6ef5",
                         label="Heuristic 95% interval")
    axes[0].axhline(truth, linestyle="--", color="#343a40", label="Reference score")
    axes[0].set_ylabel("UE-FMA")
    axes[0].set_ylim(0, 66)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(x, pd.to_numeric(frame.mean_confidence), ".-", color="#2f9e44")
    axes[1].set_ylabel("Mean confidence")
    axes[1].set_ylim(0, 1)
    withheld = [(r["status_counts"]["escalate"] + r["status_counts"]["block"]) / r["n_windows"]
                if r["n_windows"] else np.nan for r in records]
    axes[2].plot(x, withheld, ".-", color="#f08c00")
    flagged = np.asarray(frame.clinician_flag, dtype=bool)
    axes[2].scatter(x[flagged], np.asarray(withheld)[flagged], marker="x", color="#c92a2a", label="Clinician review")
    axes[2].set_ylabel("Withheld fraction")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Exercise in replay order")
    axes[2].legend(fontsize=8)
    if preview:
        for axis in axes:
            axis.axvline(x[-1], linestyle=":", color="#c92a2a", label="Scratch preview")
        axes[0].set_title("Last point is a scratch preview; persistent memory is unchanged")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def export_demo(output: Path, source, records: list[dict], patient, persistent,
                config, prompt: str, preview: bool) -> None:
    """Called after exercise close; ground truth is used only for diagnostics."""
    (output / "model_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (output / "patient_memory.json").write_text(persistent.model_dump_json(indent=2), encoding="utf-8")
    preview_path = output / "preview_patient_memory.json"
    if preview:
        preview_path.write_text(patient.model_dump_json(indent=2), encoding="utf-8")
    elif preview_path.exists():
        preview_path.unlink()
    (output / "run_details.json").write_text(json.dumps({
        "preview": preview,
        "patient_memory": "patient_memory.json",
        "result_memory": "preview_patient_memory.json" if preview else "patient_memory.json",
        "persistent_memory_updated": not preview,
    }, indent=2) + "\n", encoding="utf-8")
    write_records(records, output / "window_decisions.csv")
    history = [record.model_dump() for record in patient.exercises]
    write_records(history, output / "exercise_history.csv")
    timeline = output / "decision_timeline.png"
    if records:
        frame = pd.DataFrame(records)
        primitives = metrics.primitive_frame(frame, source)
        if not primitives.empty:
            decision_timeline(frame, primitives, timeline)
    elif timeline.exists():
        timeline.unlink()
    trajectory = output / "patient_trajectory.png"
    if len(history) > 1:
        patient_trajectory(history, config, source.true_uefma(patient.profile.subject_id), trajectory, preview)
    elif trajectory.exists():
        trajectory.unlink()
