"""Read demo exports for display without importing agents or changing saved data."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from demo_ui.timeline import final_timeline

PREDICTION_COLUMNS = [
    "Horizon", "Frame", "Perception", "Confidence", "Planning", "Belief",
    "Final", "Final confidence", "Released",
]


def _number(value: Any) -> str:
    try:
        number = float(value)
        return f"{number:.2f}" if math.isfinite(number) else "Unavailable"
    except (ValueError, TypeError):
        return "Unavailable"


def _released(value: Any) -> bool:
    return str(value).lower() in ("true", "1")


def _array(row: dict, key: str) -> list:
    value = row.get(key, "")
    parsed = json.loads(value) if value else []
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list in {key}")
    return parsed


def inspect_window(windows: list[dict], selected: str | None) -> tuple[list[list], str]:
    if selected is None or not windows:
        return [], "No window decisions available."
    row = next((item for item in windows if str(item.get("window_index")) == str(selected)), None)
    if row is None:
        return [], "Select a window from this run."
    try:
        model, plan, final = [_array(row, key) for key in ("model_action", "plan_action", "final_action")]
        model_conf, plan_conf, final_conf = [
            _array(row, key) for key in ("model_confidence", "plan_confidence", "final_confidence")
        ]
        rules = _array(row, "rule_hits")
        first_frame = int(row["target_frame_index"])
        def at(values: list, index: int, numeric: bool = False) -> Any:
            value = values[index] if index < len(values) else None
            return _number(value) if numeric else (value if value is not None else "Unavailable")
        predictions = [
            [h, first_frame + h, at(model, h), at(model_conf, h, True),
             at(plan, h), at(plan_conf, h, True), at(final, h), at(final_conf, h, True),
             "Yes" if _released(row.get("released")) else "No"]
            for h in range(max(len(model), len(plan), len(final)))
        ]
        details = (
            f"Window {selected} · first target frame {first_frame}\n"
            f"Decision: {row.get('status', 'Unavailable')} · "
            f"Selected agent: {row.get('selected_agent', 'Unavailable')}\n"
            f"Safety verdict: {row.get('safety_verdict', 'Unavailable')}\n"
            f"Rules: {'; '.join(str(rule) for rule in rules) or 'None'}\n"
            f"Impairment band: {row.get('band', 'Unavailable')}\n"
            f"Provisional UE-FMA: {_number(row.get('estimated_uefma'))}\n"
            f"Model UE-FMA (raw): {_number(row.get('model_ue_fma'))}\n"
            f"Patient transition weight: {_number(row.get('patient_weight'))}\n"
            f"Feedback: {row.get('feedback', '')}"
        )
        return predictions, details
    except (ValueError, TypeError, KeyError) as error:
        return [], f"Cannot inspect this saved window: {error}"


@dataclass
class DemoOutput:
    summary: str
    clinician: str
    technical: str
    timeline: Any
    trajectory: Any
    windows: list[dict]
    notes: str


def load_output(directory: Path, sample_id: str) -> DemoOutput:
    warnings = []

    def text_file(name: str) -> str:
        path = directory / name
        if not path.is_file():
            warnings.append(f"{name} was not exported.")
            return "Unavailable for this run."
        return path.read_text(encoding="utf-8")

    def csv_file(name: str) -> list[dict]:
        path = directory / name
        if not path.is_file():
            warnings.append(f"{name} was not exported.")
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def image_file(name: str):
        path = directory / name
        if not path.is_file():
            return None
        try:
            with Image.open(path) as image:
                return image.copy()  # Serve only these image pixels, not the export directory.
        except OSError as error:
            warnings.append(f"Could not open {name}: {error}")
            return None

    windows = csv_file("window_decisions.csv")
    windows.sort(key=lambda row: int(row["window_index"]))
    history = csv_file("exercise_history.csv")
    matches = [row for row in history if row.get("clip_id") == sample_id]
    exercise = matches[-1] if matches else {}
    first = windows[0] if windows else {}
    try:
        count = int(exercise.get("n_windows", len(windows)))
        released = int(exercise.get("released_windows", sum(_released(row.get("released")) for row in windows)))
        coverage = f"{100 * released / count:.1f}%" if count else "Unavailable"
    except (ValueError, TypeError):
        count, coverage = len(windows), "Unavailable"
    details = {}
    try:
        details = json.loads((directory / "run_details.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.append(f"Preview/memory status unavailable: {error}")
    if details.get("preview"):
        memory = "Scratch preview — persistent patient memory was not updated."
    elif details.get("persistent_memory_updated"):
        memory = "Patient memory updated in the selected SQLite database."
    else:
        memory = "Patient-memory update status unavailable."
    summary = (
        f"**Patient:** {first.get('subject_id', sample_id.split('_', 1)[0])} · "
        f"**Task:** {first.get('task_name', exercise.get('task_name', 'Unavailable'))} · "
        f"**Clip:** `{sample_id}`\n\n"
        f"| Windows | Released | Exercise UE-FMA | Pooled UE-FMA |\n"
        f"|---|---|---|---|\n"
        f"| {count} | {coverage} | {_number(exercise.get('ue_fma_exercise'))} | "
        f"{_number(exercise.get('ue_fma_estimate_after'))} |\n\n{memory}"
    )
    clinician, technical = text_file("report.txt"), text_file("report_details.txt")
    timeline = None
    try:
        timeline = final_timeline(windows)
    except (ValueError, TypeError, KeyError) as error:
        warnings.append(f"Could not draw the final-output timeline: {error}")
    trajectory = image_file("patient_trajectory.png")
    if timeline is None:
        warnings.append("No decision timeline is available; the clip may have no complete input windows.")
    if trajectory is None:
        warnings.append("No patient trajectory was exported; it requires multiple exercise records.")
    if not windows:
        warnings.append("No window decisions to inspect.")
    return DemoOutput(summary, clinician, technical, timeline, trajectory, windows, "\n\n".join(warnings))
