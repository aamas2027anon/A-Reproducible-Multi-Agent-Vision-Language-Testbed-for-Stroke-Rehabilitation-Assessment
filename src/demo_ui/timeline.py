"""Compact output-only timeline for the UI, drawn from saved window decisions."""
from __future__ import annotations

import io
import json

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

PRIMITIVE_COLORS = {
    "reach": "#4c6ef5",
    "reposition": "#f08c00",
    "rest": "#adb5bd",
    "stabilize": "#2f9e44",
    "transport": "#0891b2",
}
STATUS_COLORS = {
    "approve": "#16a34a",
    "revise": "#eab308",
    "escalate": "#ea580c",
    "block": "#475569",
}


def _append_span(spans: list, start: int, end: int, label: str) -> None:
    if spans and spans[-1][1] == start and spans[-1][2] == label:
        spans[-1] = (spans[-1][0], end, label)
    else:
        spans.append((start, end, label))


def final_timeline(windows: list[dict]) -> Image.Image | None:
    """Show final primitives and verdicts as two thin strips on the frame axis."""
    final_spans, status_spans, withheld = [], [], []
    for row in windows:
        start = int(row["target_frame_index"])
        actions = json.loads(row["final_action"])
        if not isinstance(actions, list):
            raise ValueError("Final actions must be a list of primitive labels.")
        if not actions:
            continue
        end = start + len(actions)
        for horizon, action in enumerate(actions):
            _append_span(final_spans, start + horizon, start + horizon + 1, action)
        _append_span(status_spans, start, end, row["status"])
        if str(row.get("released", "")).lower() not in ("true", "1"):
            withheld.append((start, end - start))
    if not final_spans:
        return None

    figure = Figure(figsize=(11, 1.8), dpi=160, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    figure.subplots_adjust(left=0.075, right=0.99, top=0.68, bottom=0.24, hspace=0.15)
    for axis, spans, colors, label in zip(
        axes, (final_spans, status_spans), (PRIMITIVE_COLORS, STATUS_COLORS), ("Final", "Status"),
    ):
        for start, end, name in spans:
            axis.broken_barh([(start, end - start)], (0.35, 0.3),
                            facecolors=colors.get(name, "#94a3b8"), edgecolors="none")
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, labelpad=10)
        for side in ("top", "left", "right"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#e2e8f0")
        axis.tick_params(axis="x", labelsize=8, colors="#475569", length=3)
    if withheld:
        axes[0].broken_barh(withheld, (0.35, 0.3), facecolors=(1, 1, 1, 0.6),
                            edgecolors="#64748b", linewidths=0, hatch="///")
    axes[0].spines["bottom"].set_visible(False)
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[1].set_xlim(final_spans[0][0], final_spans[-1][1])
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axes[1].set_xlabel("Frame index", fontsize=9, color="#475569", labelpad=3)
    primitive_legend = [Patch(facecolor=color, label=name.capitalize())
                        for name, color in PRIMITIVE_COLORS.items()]
    primitive_legend.append(Patch(facecolor="white", edgecolor="#64748b", hatch="///", label="Withheld"))
    figure.legend(handles=primitive_legend, loc="upper left", bbox_to_anchor=(0.07, 1.0),
                  ncol=6, frameon=False, fontsize=8)
    figure.legend(handles=[Patch(facecolor=color, label=name.capitalize())
                           for name, color in STATUS_COLORS.items()],
                  loc="upper left", bbox_to_anchor=(0.07, 0.85), ncol=4, frameon=False, fontsize=8)
    with io.BytesIO() as buffer:
        canvas.print_png(buffer)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return image.copy()
