"""The coordinator's report templates."""

from __future__ import annotations

# The report a clinician reads: short, plain language, patient-specific. Every field
# is filled by code from the agents' results.
CLINICIAN_REPORT_TEMPLATE = """\
Patient {subject_id} · exercise {exercise_number} · {task_label} task
Movement: {movement}
Arm function (UE-FMA): {function}
Safety: {safety}
Clinician review: {review}
Summary: {summary}"""

# Technical detail of the same exercise (report_details.txt), for developers and audits.
SYSTEM_DETAILS_TEMPLATE = """\
=== System details: {subject_id} exercise #{exercise_number} ({clip_id}) ===
Decision: {decision} | released windows: {released}/{windows} ({coverage})
Window statuses: approve {approve} | revise {revise} | escalate {escalate} | block {block}
Future primitive at the end: {next_cue} (confidence {next_confidence})
Planned primitives of the last released window (current -> future): {plan}
UE-FMA: this exercise {ue_exercise}/66 | patient estimate {ue_estimate} +/- {ue_ci} (95% CI) | band: {band} | update: {ue_update}
Safety: {safety_summary}
Coordination: {disagreements} perception/planning disagreements resolved, {revisions} safety revisions released, {failures} coordination failures
Patient trend (last {trend_window} exercises): UE-FMA {trend_uefma} | mean confidence {trend_confidence} | escalation rate {trend_escalation}
Clinician flags: {clinician}"""
