"""Figures: patient trajectories, agent decisions, and MAS-vs-VLM comparisons.

Static PNGs (matplotlib, like the training pipeline's plots). Colors follow fixed
roles: primitives and modes take categorical slots in a fixed order (an entity keeps
its color in every figure), decision statuses use the reserved status palette and
always carry a text label, and all text stays in ink colors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from multi_agents.settings import HORIZONS, PRIMITIVES, STREAM_FPS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b3"
GRID_HIT = "#dcdbd6"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
STATUS = {"approve": "#0ca30c", "revise": "#fab219", "escalate": "#ec835a", "block": "#d03b3b"}
MODE_ORDER = ("vlm", "mas", "mas_no_memory", "mas_no_planning", "mas_no_safety")
MODE_LABELS = {
    "vlm": "VLM (single model)",
    "mas": "MAS",
    "mas_no_memory": "MAS, no memory",
    "mas_no_planning": "MAS, no planning",
    "mas_no_safety": "MAS, no safety",
}
# Rest means no movement, so it is drawn in neutral grey; only active primitives get hues
# (this also keeps rest from blending into the green "approve" status).
REST_COLOR = "#8f8e89"
PRIMITIVE_COLORS = {
    name: REST_COLOR if name == "rest" else CATEGORICAL[i] for i, name in enumerate(PRIMITIVES)
}
PRIMITIVE_PALETTE = [PRIMITIVE_COLORS[name] for name in PRIMITIVES]
AGENT_ORDER = ("perception_model", "perception", "planning", "safety", "evaluation", "coordinator")


def mode_color(mode: str) -> str:
    return CATEGORICAL[MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER)]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "text.color": INK,
            "lines.linewidth": 2,
        }
    )


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _modes(frame: pd.DataFrame) -> list[str]:
    present = list(frame["mode"])
    return [m for m in MODE_ORDER if m in present] + [m for m in present if m not in MODE_ORDER]


# ---------------------------------------------------------------- comparisons

COMPARISON_PANELS = (
    ("accuracy_future_primitive", "Future-primitive accuracy (the recommendation)", "{:.3f}"),
    ("step_ms_p95", f"Per-step latency p95, ms (real-time budget {5000 / STREAM_FPS:.0f} ms)", "{:.1f}"),
    ("accuracy_all", "Primitive accuracy (all 5 frames)", "{:.3f}"),
    ("macro_f1_all", "Primitive macro-F1", "{:.3f}"),
    ("count_mae", "Primitive count MAE per subject (lower is better)", "{:.1f}"),
    ("ue_fma_exercise_mae", "UE-FMA MAE per exercise (lower is better)", "{:.2f}"),
    ("unsafe_recommendation_rate", "Unsafe recommendation rate (lower is better)", "{:.4f}"),
    ("unnecessary_escalation_rate", "Unnecessary escalation rate (lower is better)", "{:.3f}"),
    ("coverage_windows", "Coverage: windows with a released cue", "{:.3f}"),
    ("coordination_failure_rate", "Coordination failures per window", "{:.4f}"),
)


def comparison_bars(comparison: pd.DataFrame, path: Path) -> Path:
    _style()
    modes = _modes(comparison)
    frame = comparison.set_index("mode").loc[modes]
    figure, axes = plt.subplots(5, 2, figsize=(11, 12.5))
    for axis, (metric, title, fmt) in zip(axes.flat, COMPARISON_PANELS):
        values = frame[metric].astype(float).to_numpy()
        positions = np.arange(len(modes))
        axis.barh(
            positions, values, height=0.6, color=[mode_color(m) for m in modes], edgecolor=SURFACE, linewidth=2
        )
        axis.set_yticks(positions, [MODE_LABELS.get(m, m) for m in modes])
        axis.invert_yaxis()
        axis.set_title(title, loc="left")
        axis.grid(axis="y", visible=False)
        span = np.nanmax(values) if np.isfinite(values).any() else 1.0
        for position, value in zip(positions, values):
            if np.isfinite(value):
                axis.text(value + 0.01 * (span or 1), position, fmt.format(value), va="center", fontsize=8, color=INK)
        axis.set_xlim(0, (span or 1) * 1.18)
    figure.suptitle("Single VLM vs multi-agent system (test subjects)", x=0.01, ha="left", fontsize=13)
    figure.tight_layout()
    return _save(figure, path)


# Segment-sequence metrics (multi_agents/sequence_metrics.py): (column, label, higher is better).
SEQUENCE_PANELS = (
    ("edit_score", "Edit score", True),
    ("action_error_rate", "Action error rate", False),
    ("relative_counting_error", "Relative counting error", False),
)
SEQUENCE_LEVELS = (("frame", "every frame"), ("window", "current frame of each window"))


def _short_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode).replace(" (single model)", "").replace(", ", "\n")


def _clip_boxes(axis, groups: list[np.ndarray], modes: list[str], fmt: str) -> None:
    """One box per mode over the clips (quartiles, mean marker, no outlier dots); the mean is in the tick label."""
    positions = np.arange(1, len(groups) + 1)
    boxes = axis.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        showmeans=True,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.2},
        meanprops={"marker": "D", "markerfacecolor": SURFACE, "markeredgecolor": INK, "markersize": 5},
        whiskerprops={"color": INK_SECONDARY, "linewidth": 1},
        capprops={"color": INK_SECONDARY, "linewidth": 1},
    )
    for patch, mode in zip(boxes["boxes"], modes):
        patch.set(facecolor=mode_color(mode), alpha=0.8, edgecolor=SURFACE, linewidth=1.5)
    labels = [
        f"{_short_label(mode)}\n{fmt.format(np.mean(values))}" if len(values) else _short_label(mode)
        for mode, values in zip(modes, groups)
    ]
    axis.set_xticks(positions, labels, fontsize=7.5)
    axis.grid(axis="x", visible=False)


def sequence_metrics(clips_by_mode: dict[str, pd.DataFrame], path: Path) -> Path:
    """Edit score, action error rate and relative counting error per clip, one box per mode."""
    _style()
    modes = [m for m in MODE_ORDER if m in clips_by_mode] + [m for m in clips_by_mode if m not in MODE_ORDER]
    figure, axes = plt.subplots(len(SEQUENCE_LEVELS), len(SEQUENCE_PANELS), figsize=(16, 9.5), squeeze=False)
    for row, (level, level_text) in enumerate(SEQUENCE_LEVELS):
        for column, (metric, label, higher) in enumerate(SEQUENCE_PANELS):
            axis = axes[row, column]
            groups = []
            for mode in modes:
                clips = clips_by_mode[mode]
                selected = clips.loc[clips["level"] == level, metric] if len(clips) else pd.Series(dtype=float)
                groups.append(selected.dropna().to_numpy(dtype=float))
            _clip_boxes(axis, groups, modes, "mean {:.3f}")
            axis.set_title(f"{label}, {level_text} ({'higher' if higher else 'lower'} is better)", loc="left", fontsize=10)
            if column == 0:
                axis.set_ylabel("Per clip")
    figure.suptitle("Segment-sequence metrics per clip (box = quartiles, diamond = mean)", x=0.01, ha="left", fontsize=13)
    figure.tight_layout()
    return _save(figure, path)


def sequence_metrics_vs_vlm(clips_by_mode: dict[str, pd.DataFrame], path: Path) -> Path | None:
    """Per clip, each MAS mode minus the VLM on the same clip: where the agents gain or lose."""
    if "vlm" not in clips_by_mode or len(clips_by_mode["vlm"]) == 0:
        return None
    _style()
    keys = ["subject_id", "clip_id", "level"]
    base = clips_by_mode["vlm"]
    modes = [m for m in MODE_ORDER if m in clips_by_mode and m != "vlm"] + [
        m for m in clips_by_mode if m not in MODE_ORDER and m != "vlm"
    ]
    if not modes:
        return None
    figure, axes = plt.subplots(len(SEQUENCE_LEVELS), len(SEQUENCE_PANELS), figsize=(16, 9.5), squeeze=False)
    for row, (level, level_text) in enumerate(SEQUENCE_LEVELS):
        for column, (metric, label, higher) in enumerate(SEQUENCE_PANELS):
            axis = axes[row, column]
            groups = []
            for mode in modes:
                paired = clips_by_mode[mode].merge(base, on=keys, suffixes=("", "_vlm"))
                paired = paired[paired["level"] == level]
                groups.append((paired[metric] - paired[f"{metric}_vlm"]).dropna().to_numpy(dtype=float))
            _clip_boxes(axis, groups, modes, "mean Δ {:+.3f}")
            axis.axhline(0, color=INK_SECONDARY, linewidth=1, linestyle="--", zorder=1)
            better = "above" if higher else "below"
            axis.set_title(f"Δ {label}, {level_text} (better: {better} 0)", loc="left", fontsize=10)
            if column == 0:
                axis.set_ylabel("Mode − VLM, same clip")
    figure.suptitle("Segment-sequence metrics: change vs the single VLM, paired per clip", x=0.01, ha="left", fontsize=13)
    figure.tight_layout()
    return _save(figure, path)


def safety_tradeoff(comparison: pd.DataFrame, path: Path) -> Path:
    _style()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for mode in _modes(comparison):
        row = comparison[comparison["mode"] == mode].iloc[0]
        x, y = row["unnecessary_escalation_rate"], row["unsafe_recommendation_rate"]
        axis.scatter(x, y, s=70, color=mode_color(mode), edgecolor=SURFACE, linewidth=2, zorder=3)
        axis.annotate(MODE_LABELS.get(mode, mode), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    axis.set_xlabel("Unnecessary escalation rate (share of withheld frames)")
    axis.set_ylabel("Unsafe recommendation rate (per released frame)")
    axis.set_title("Safety trade-off: lower-left is better", loc="left")
    return _save(figure, path)


def latency_breakdown(comparison: pd.DataFrame, path: Path) -> Path:
    _style()
    modes = _modes(comparison)
    figure, axis = plt.subplots(figsize=(8, 0.6 * len(modes) + 1.6))
    left = np.zeros(len(modes))
    for slot, agent in enumerate(AGENT_ORDER):
        column = f"latency_ms_per_window_{agent}"
        if column not in comparison:
            continue
        values = comparison.set_index("mode").loc[modes, column].fillna(0.0).to_numpy(float)
        axis.barh(
            np.arange(len(modes)), values, left=left, height=0.6, color=CATEGORICAL[slot],
            edgecolor=SURFACE, linewidth=2, label=agent.replace("_", " "),
        )
        left += values
    for position, total in enumerate(left):
        axis.text(total, position, f"  {total:.1f} ms", va="center", fontsize=8, color=INK)
    axis.set_yticks(np.arange(len(modes)), [MODE_LABELS.get(m, m) for m in modes])
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("Wall-clock ms per window")
    axis.set_title("Computation per window by agent", loc="left")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return _save(figure, path)


def risk_coverage(primitives_by_mode: dict[str, pd.DataFrame], path: Path) -> Path:
    """Selective accuracy when only the most confident frames are cued."""
    _style()
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for mode in [m for m in MODE_ORDER if m in primitives_by_mode]:
        frame = primitives_by_mode[mode].sort_values("prediction_probability", ascending=False)
        correct = (frame["actual_label_id"] == frame["predicted_label_id"]).to_numpy(float)
        coverage = np.arange(1, len(correct) + 1) / len(correct)
        accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        step = max(1, len(correct) // 400)
        axis.plot(coverage[::step], accuracy[::step], color=mode_color(mode), label=MODE_LABELS.get(mode, mode))
    axis.set_xlabel("Coverage (share of frames cued, most confident first)")
    axis.set_ylabel("Accuracy of cued frames")
    axis.set_title("Risk-coverage", loc="left")
    axis.legend()
    return _save(figure, path)


# ---------------------------------------------------------------- trajectories


def patient_trajectory(exercises_by_mode: dict[str, pd.DataFrame], subject_id: str, path: Path, mas_mode: str = "mas") -> Path:
    """UE-FMA estimate, accuracy and agent decisions across a patient's exercises."""
    _style()
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(10, 9), sharex=True, height_ratios=[3, 2, 2])
    vlm = exercises_by_mode.get("vlm")
    mas = exercises_by_mode.get(mas_mode)
    vlm = None if vlm is None else vlm[vlm["subject_id"] == subject_id].sort_values("exercise_index")
    mas = None if mas is None else mas[mas["subject_id"] == subject_id].sort_values("exercise_index")
    reference = mas if mas is not None and len(mas) else vlm
    x = reference["exercise_index"].to_numpy() + 1

    top.axhline(reference["ue_true"].iloc[0], color=INK_SECONDARY, linestyle="--", linewidth=1.2, label="Clinical UE-FMA (reference)")
    if vlm is not None and len(vlm):
        top.scatter(vlm["exercise_index"] + 1, vlm["ue_model_exercise"], s=18, color=mode_color("vlm"), label="VLM: this exercise", zorder=3)
        top.plot(vlm["exercise_index"] + 1, vlm["ue_model_running_mean"], color=mode_color("vlm"), linewidth=1.2, alpha=0.6, label="VLM: running mean")
    if mas is not None and len(mas):
        estimate = mas["ue_estimate"].to_numpy(float)
        ci = mas["ue_ci95"].to_numpy(float)
        top.plot(mas["exercise_index"] + 1, estimate, color=mode_color(mas_mode), label=f"{MODE_LABELS[mas_mode]}: patient estimate")
        top.fill_between(mas["exercise_index"] + 1, estimate - ci, estimate + ci, color=mode_color(mas_mode), alpha=0.15, linewidth=0, label="95% CI")
        flagged = mas[mas["clinician_flag"].astype(bool)]
        if len(flagged):
            top.scatter(flagged["exercise_index"] + 1, flagged["ue_estimate"], marker="^", s=60, color=STATUS["escalate"], edgecolor=SURFACE, linewidth=1.5, zorder=4, label="Clinician review requested")
    top.set_ylabel("UE-FMA (0-66)")
    top.set_title(f"Patient {subject_id}: repeated exercises", loc="left")
    top.legend(ncol=2, loc="lower right")

    for frame, mode in ((vlm, "vlm"), (mas, mas_mode)):
        if frame is not None and len(frame):
            middle.plot(frame["exercise_index"] + 1, frame["accuracy"], marker="o", markersize=4, color=mode_color(mode), label=MODE_LABELS.get(mode, mode))
    middle.set_ylabel("Primitive accuracy")
    middle.legend(loc="lower right")

    if mas is not None and len(mas):
        base = np.zeros(len(mas))
        for status in ("approve", "revise", "escalate", "block"):
            share = mas[status].to_numpy(float) / mas["n_windows"].to_numpy(float)
            bottom.bar(mas["exercise_index"] + 1, share, bottom=base, color=STATUS[status], edgecolor=SURFACE, linewidth=1, label=status)
            base += share
        bottom.set_ylim(0, 1)
        bottom.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    bottom.set_ylabel("Share of windows")
    bottom.set_xlabel("Exercise number (pseudo-session order)")
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
    bottom.grid(axis="x", visible=False)
    if len(x):
        bottom.set_xlim(x.min() - 0.6, x.max() + 0.6)
    figure.tight_layout()
    return _save(figure, path)


def adaptation_curve(exercises_by_mode: dict[str, pd.DataFrame], path: Path) -> Path:
    """Mean over patients against exercise number: does memory help as history grows?"""
    _style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for mode in [m for m in MODE_ORDER if m in exercises_by_mode]:
        frame = exercises_by_mode[mode]
        grouped = frame.groupby("exercise_index")
        index = np.asarray(sorted(frame["exercise_index"].unique())) + 1
        left.plot(index, grouped["accuracy"].mean().to_numpy(), color=mode_color(mode), label=MODE_LABELS.get(mode, mode))
        right.plot(index, grouped["ue_abs_error_estimate"].mean().to_numpy(), color=mode_color(mode), label=MODE_LABELS.get(mode, mode))
    if "vlm" in exercises_by_mode:
        vlm = exercises_by_mode["vlm"].groupby("exercise_index")["ue_abs_error_model_running_mean"].mean()
        right.plot(vlm.index + 1, vlm.to_numpy(), color=mode_color("vlm"), linestyle=":", linewidth=1.5, label="VLM running mean")
    left.set_title("Primitive accuracy by exercise number", loc="left")
    left.set_xlabel("Exercise number")
    left.set_ylabel("Mean accuracy across patients")
    right.set_title("UE-FMA absolute error by exercise number", loc="left")
    right.set_xlabel("Exercise number")
    right.set_ylabel("Mean |estimate - reference|")
    left.legend()
    right.legend()
    for axis in (left, right):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    return _save(figure, path)


def decision_timeline(
    records: pd.DataFrame,
    primitives: pd.DataFrame,
    clip_id: str,
    path: Path,
    subject_id: str | None = None,
) -> Path:
    """What the system recommended, drawn on the frames each recommendation is about.

    A window completes at frame t (its last context frame) and predicts frames t..t+4.
    Its "now" prediction (h0) is about frame t and holds until the next window, so it is
    drawn over [t, t+stride). Its "next" prediction (h4) is about frame t+4 and is drawn
    over [t+4, t+4+stride), exactly under the ground truth it should match. The next
    window's t+4 prediction takes over at t+4+stride, so each row is a gap-free strip.
    """
    _style()
    in_clip = lambda frame: (frame["clip_id"] == clip_id) & (  # noqa: E731
        True if subject_id is None else frame["subject_id"] == subject_id
    )
    clip = primitives[in_clip(primitives)].sort_values("absolute_frame_index")
    windows = records[in_clip(records)].sort_values("window_index")
    targets = windows["target_frame_index"].to_numpy(dtype=int)
    stride = int(np.median(np.diff(targets))) if len(targets) > 1 else HORIZONS
    lead = HORIZONS - 1  # the next prediction is about frame t + 4
    truth = dict(zip(clip["absolute_frame_index"].astype(int), clip["actual_label_id"].astype(int)))
    index = {name: i for i, name in enumerate(PRIMITIVES)}

    # Current group, then future group (each read against the ground truth directly
    # above it), then the decision per window. `None` entries are gaps between groups.
    rows = [
        ("Ground truth", "truth"),
        ("VLM current", ("model_action", 0)),
        ("MAS current", ("final_action", 0)),
        None,
        ("Ground truth", "truth"),
        ("VLM future", ("model_action", lead)),
        ("MAS future", ("final_action", lead)),
        ("Future correct?", "correct"),
        None,
        ("Decision", "status"),
    ]
    gap = 0.5
    positions, y = [], 0.0
    for row in reversed(rows):
        if row is None:
            y += gap
            continue
        positions.append((y, row))
        y += 1.0
    positions.reverse()
    figure, (strips, confidence) = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True, height_ratios=[3.8, 1.4])
    height = 0.8
    for y, (_, kind) in positions:
        if kind == "truth":
            frames = sorted(truth)
            strips.broken_barh([(f, 1) for f in frames], (y, height), facecolors=[PRIMITIVE_PALETTE[truth[f]] for f in frames])
            continue
        spans, colors, hatched = [], [], []
        for window in windows.itertuples(index=False):
            t = int(window.target_frame_index)
            if kind == "status":
                spans.append((t, stride))
                colors.append(STATUS[window.status])
            elif kind == "correct":
                frame = t + lead
                if frame not in truth:
                    continue
                spans.append((frame, stride))
                # Hits recede in neutral grey; only misses are coloured.
                colors.append(GRID_HIT if index[window.final_action[lead]] == truth[frame] else STATUS["block"])
            else:
                column, horizon = kind
                start = t + horizon
                spans.append((start, stride))
                colors.append(PRIMITIVE_PALETTE[index[getattr(window, column)[horizon]]])
                if column == "final_action" and not window.released:
                    hatched.append((start, stride))
        strips.broken_barh(spans, (y, height), facecolors=colors, linewidth=0)
        if hatched:
            strips.broken_barh(hatched, (y, height), facecolors="none", hatch="////", edgecolor=SURFACE, linewidth=0)

    strips.set_yticks([y + height / 2 for y, _ in positions], [name for _, (name, _) in positions])
    strips.grid(False)
    handles = [Patch(color=PRIMITIVE_COLORS[p], label=p) for p in PRIMITIVES]
    handles += [Patch(color=STATUS[s], label=f"decision: {s}") for s in STATUS]
    handles += [
        Patch(color=GRID_HIT, label="future: correct"),
        Patch(color=STATUS["block"], label="future: wrong"),
        Patch(facecolor=NEUTRAL, hatch="////", edgecolor=SURFACE, label="cue withheld"),
    ]
    strips.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    next_rows = clip[clip["horizon"] == lead]
    accuracy_note = ""
    if len(next_rows):
        vlm_next = (next_rows["model_label_id"] == next_rows["actual_label_id"]).mean()
        mas_next = (next_rows["predicted_label_id"] == next_rows["actual_label_id"]).mean()
        accuracy_note = f" | future correct: VLM {vlm_next:.0%}, MAS {mas_next:.0%}"
    task = windows["task_name"].iloc[0] if len(windows) else ""
    strips.set_title(f"{clip_id} ({task}): recommendations on the frames they predict{accuracy_note}", loc="left", pad=62)

    now_frames = windows["target_frame_index"].to_numpy(dtype=int)
    confidence.plot(now_frames, [c[0] for c in windows["model_confidence"]], color=mode_color("vlm"), label="VLM current")
    confidence.plot(now_frames, [c[0] for c in windows["final_confidence"]], color=mode_color("mas"), label="MAS current")
    confidence.plot(
        now_frames + lead,
        [c[lead] for c in windows["final_confidence"]],
        color=mode_color("mas"),
        linestyle="--",
        linewidth=1.5,
        label="MAS future (shown at the frame it predicts)",
    )
    confidence.set_ylim(0, 1.02)
    confidence.set_ylabel("Confidence")
    confidence.set_xlabel("Frame")
    confidence.legend(loc="lower right", ncol=3)
    if len(targets):
        confidence.set_xlim(targets.min() - 1, targets.max() + lead + stride + 1)
    figure.tight_layout()
    return _save(figure, path)
