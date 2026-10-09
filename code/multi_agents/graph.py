"""The decision cycle as an event-driven LangGraph state machine.

A live stream drives the graph: every invocation runs one event on the patient's
thread, and the checkpointer carries the state between invocations.

    event=start   start_exercise                  coordinator resets per-exercise state,
                                                  perception opens the (live or recorded) exercise
    event=window  observe                         perception scores the newest window   (observe patient)
                  -> propose                      perception + planning proposals,      (agents propose)
                                                  evaluation accumulates UE-FMA
                  -> safety_check                 safety rules R1, R2, C1               (check safety)
                  -> coordinate                   release safety's revision or arbitrate (coordinate)
                  -> deliver                      release or withhold the cue, update   (deliver, update
                                                  the patient's primitive memory         patient memory)
    event=end     assess                          evaluation updates the patient UE-FMA estimate
                  -> exercise_review              safety's exercise-level rules
                  -> report                       coordinator fills its template, appends ExerciseRecord
                  -> commit_memory                close the exercise

`multi_agents.session.RehabSession` sends the events: one `window` event per
completed window (every 5 frames of a live stream), so each recommendation is
produced as soon as its window is complete.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from langgraph.graph import END, START, StateGraph

from multi_agents.agents.coordinator import CoordinatorAgent
from multi_agents.agents.evaluation import EvaluationAgent
from multi_agents.agents.perception import PerceptionAgent
from multi_agents.agents.planning import PlanningAgent
from multi_agents.agents.safety import SafetyAgent, plain_reason
from multi_agents.memory import WindowLog
from multi_agents.settings import PRIMITIVES, VERDICTS, MASConfig
from multi_agents.state import (
    ExerciseContext,
    ExerciseRecord,
    PerceptionReport,
    PatientMemory,
    PatientProfile,
    Proposal,
    RehabState,
    SafetyVerdict,
    WindowDecision,
    assert_no_ground_truth,
)

AGENTS = ("perception", "planning", "safety", "evaluation", "coordinator")
TREND_WINDOW = 3


@dataclass
class Services:
    """Runtime objects the nodes use; never part of the checkpointed state."""

    config: MASConfig
    perception: PerceptionAgent
    planning: PlanningAgent
    safety: SafetyAgent
    evaluation: EvaluationAgent
    coordinator: CoordinatorAgent
    log: WindowLog
    mode: str
    reports: list[str] = field(default_factory=list)


def new_stats() -> dict[str, Any]:
    return {
        "window_scores": [],
        "status_counts": {verdict: 0 for verdict in VERDICTS},
        "released": 0,
        "disagreements": 0,
        "revisions": 0,
        "coordination_failures": 0,
        "confidences": [],
        "predicted_counts": {name: 0 for name in PRIMITIVES},
        "band": None,
        "estimate": None,
        "usable_windows": 0,
        "window_started": None,
        "latency_ms": {agent: 0.0 for agent in (*AGENTS, "perception_model")},
    }


@contextmanager
def timed(stats: dict[str, Any], agent: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        stats["latency_ms"][agent] += 1000.0 * (time.perf_counter() - started)


def _bump(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def build_graph(services: Services, checkpointer):
    svc, config = services, services.config

    def start_exercise(state: RehabState) -> dict:
        stats = new_stats()
        with timed(stats, "coordinator"):
            exercise = ExerciseContext.model_validate(state["exercise"])
            patient = state.get("patient")
            if patient is None:
                seed = state["patient_seed"]
                patient = PatientMemory(
                    profile=PatientProfile.model_validate(seed["profile"]),
                    config_hash=seed.get("config_hash"),
                    checkpoint=seed.get("checkpoint"),
                ).model_dump()
        with timed(stats, "perception"):
            svc.perception.open(exercise.key, exercise.streaming)
        exercise = exercise.model_copy(update={"n_windows": 0, "exercise_index": len(patient["exercises"])})
        # A new exercise starts from rest: no transition carries over from the previous
        # clip (it would be spurious in the patient's transition counts).
        patient = {**patient, "primitive": {**patient["primitive"], "last_released": None}}
        return {
            "patient": patient,
            "exercise": exercise.model_dump(),
            "cursor": 0,
            "report": None,
            "belief": None,
            "proposals": {},
            "verdict": None,
            "decision": None,
            "stats": stats,
            "blocked_reason": None,
            "exercise_report": None,
        }

    def observe(state: RehabState) -> dict:
        stats = state["stats"]
        stats["window_started"] = time.perf_counter()
        with timed(stats, "perception"):
            report = svc.perception.observe(state["exercise"]["key"])
        # Model compute is reported separately; the agent bucket keeps only its own overhead.
        stats["latency_ms"]["perception_model"] += svc.perception.last_model_ms
        stats["latency_ms"]["perception"] -= svc.perception.last_spent_ms
        stats["usable_windows"] += int(report.integrity_ok)
        return {"report": report.model_dump(), "stats": stats}

    def propose(state: RehabState) -> dict:
        stats = state["stats"]
        report = PerceptionReport.model_validate(state["report"])
        with timed(stats, "perception"):
            perception = svc.perception.propose(report)
        with timed(stats, "planning"):
            plan = svc.planning.plan(
                report, state.get("belief"), state["patient"]["primitive"], state["exercise"]["task_name"]
            )
        with timed(stats, "evaluation"):
            if report.integrity_ok:
                stats["window_scores"].append(report.ue_fma)
        proposals = {"perception": perception.model_dump(), "planning": plan.model_dump()}
        return {"proposals": proposals, "stats": stats}

    def safety_check(state: RehabState) -> dict:
        stats, patient = state["stats"], state["patient"]
        report = PerceptionReport.model_validate(state["report"])
        plan = Proposal.model_validate(state["proposals"]["planning"])
        with timed(stats, "evaluation"):
            estimate, band = svc.evaluation.provisional(patient["ue_fma"], stats["window_scores"])
        stats["band"], stats["estimate"] = band, estimate
        with timed(stats, "safety"):
            verdict = svc.safety.check_window(report=report, plan=plan, band=band)
        proposals = {**state["proposals"], "safety": SafetyAgent.proposal(verdict, plan).model_dump()}
        return {"verdict": verdict.model_dump(), "proposals": proposals, "stats": stats}

    def coordinate(state: RehabState) -> dict:
        stats = state["stats"]
        with timed(stats, "coordinator"):
            decision = svc.coordinator.resolve(
                report=PerceptionReport.model_validate(state["report"]),
                perception=Proposal.model_validate(state["proposals"]["perception"]),
                planning=Proposal.model_validate(state["proposals"]["planning"]),
                verdict=SafetyVerdict.model_validate(state["verdict"]),
            )
        return {"decision": decision.model_dump(), "stats": stats}

    def deliver(state: RehabState) -> dict:
        stats, patient = state["stats"], state["patient"]
        report = PerceptionReport.model_validate(state["report"])
        verdict = SafetyVerdict.model_validate(state["verdict"])
        plan = Proposal.model_validate(state["proposals"]["planning"])
        perception = Proposal.model_validate(state["proposals"]["perception"])
        exercise = state["exercise"]
        with timed(stats, "coordinator"):
            decision = WindowDecision.model_validate(state["decision"])
            decision = decision.model_copy(update={"feedback": svc.coordinator.feedback(decision, verdict)})
            primitive = dict(patient["primitive"])
            agents = {key: (dict(value) if isinstance(value, dict) else value) for key, value in patient["agents"].items()}

            if decision.released:
                counts = [list(row) for row in primitive["transition_counts"]]
                previous = primitive["last_released"]
                for cue, confidence in zip(decision.final_action, decision.confidence):
                    if confidence >= config.patient_count_min_confidence and previous is not None:
                        counts[PRIMITIVES.index(previous)][PRIMITIVES.index(cue)] += 1.0
                    previous = cue
                    stats["predicted_counts"][cue] += 1
                primitive.update(transition_counts=counts, last_released=decision.final_action[-1])
                stats["released"] += 1
                stats["confidences"].append(float(np.mean(decision.confidence)))
            else:
                primitive["last_released"] = None

            stats["status_counts"][decision.status] += 1
            stats["disagreements"] += int(decision.disagreement)
            stats["revisions"] += int(decision.status == "revise")
            stats["coordination_failures"] += int(decision.coordination_failure is not None)
            for agent in ("perception", "planning", "safety"):
                _bump(agents["proposals"], agent)
            _bump(agents["selected"], decision.selected_agent)
            _bump(agents["safety_verdicts"], verdict.verdict)
            if decision.final_action != list(plan.action):
                _bump(agents["overridden"], "planning")
            if decision.final_action != list(perception.action):
                _bump(agents["overridden"], "perception")

            svc.log.add(
                {
                    "mode": svc.mode,
                    "subject_id": exercise["subject_id"],
                    "clip_id": exercise["clip_id"],
                    "task_name": exercise["task_name"],
                    "exercise_index": exercise["exercise_index"],
                    "window_index": report.window_index,
                    "target_frame_index": report.target_frame_index,
                    "integrity_ok": report.integrity_ok,
                    "model_action": list(perception.action),
                    "model_confidence": report.confidence,
                    "model_p_rest": [row["rest"] for row in report.probs],
                    "model_ue_fma": report.ue_fma,
                    "plan_action": list(plan.action),
                    "plan_confidence": plan.details["horizon_confidence"],
                    "final_action": decision.final_action,
                    "final_confidence": decision.confidence,
                    "status": decision.status,
                    "released": decision.released,
                    "selected_agent": decision.selected_agent,
                    "disagreement": decision.disagreement,
                    "coordination_failure": decision.coordination_failure,
                    "rule_hits": verdict.rule_hits,
                    "band": stats["band"],
                    "estimated_uefma": stats["estimate"],
                    "current_primitive": decision.current_primitive,
                    "next_primitive": decision.next_primitive,
                    "next_confidence": decision.next_confidence,
                    "feedback": decision.feedback,
                    "step_ms": 1000.0 * (time.perf_counter() - stats["window_started"]),
                }
            )
        return {
            "exercise": {**exercise, "n_windows": exercise["n_windows"] + 1},
            "patient": {**patient, "primitive": primitive, "agents": agents},
            "belief": plan.details["carry"],
            "cursor": state["cursor"] + 1,
            "decision": decision.model_dump(),
            "stats": stats,
        }

    def assess(state: RehabState) -> dict:
        stats, patient, exercise = state["stats"], state["patient"], state["exercise"]
        blocked = None
        if stats["usable_windows"] == 0:
            blocked = "no usable window (non-finite input or no complete window)"
        with timed(stats, "evaluation"):
            proposal, ue_memory = svc.evaluation.assess(patient["ue_fma"], stats["window_scores"], exercise["exercise_index"])
        stats["uefma"] = {
            "exercise_score": proposal.details["exercise_score"],
            "decision": proposal.action,
            "drift": proposal.details["drift"],
        }
        return {
            "patient": {**patient, "ue_fma": ue_memory},
            "proposals": {**(state.get("proposals") or {}), "evaluation": proposal.model_dump()},
            "stats": stats,
            "blocked_reason": blocked,
        }

    def exercise_review(state: RehabState) -> dict:
        stats, patient, exercise = state["stats"], state["patient"], state["exercise"]
        windows = exercise["n_windows"]
        withheld = stats["status_counts"]["escalate"] + stats["status_counts"]["block"]
        summary = {
            "task_name": exercise["task_name"],
            "windows": windows,
            "status_counts": stats["status_counts"],
            "escalation_rate": withheld / windows if windows else 1.0,
            "blocked_reason": state.get("blocked_reason"),
            "uefma_drift": stats["uefma"]["drift"],
            "estimated_uefma": svc.evaluation.calibrated(patient["ue_fma"])[0],
            "band": patient["ue_fma"]["band"],
            "mean_confidence": float(np.mean(stats["confidences"])) if stats["confidences"] else None,
        }
        with timed(stats, "safety"):
            review = svc.safety.review_exercise(patient=patient, summary=summary)
        stats["clinician"] = review
        stats["escalation_rate"] = summary["escalation_rate"]
        return {"stats": stats}

    def report(state: RehabState) -> dict:
        stats, patient, exercise = state["stats"], state["patient"], state["exercise"]
        with timed(stats, "coordinator"):
            # Exported estimates are calibrated (mu_cal, sigma_cal); pooling stays on the raw scale.
            mean_cal, sd_cal = svc.evaluation.calibrated(patient["ue_fma"])
            record = ExerciseRecord(
                exercise_index=exercise["exercise_index"],
                clip_id=exercise["clip_id"],
                task_name=exercise["task_name"],
                n_windows=exercise["n_windows"],
                status_counts=stats["status_counts"],
                released_windows=stats["released"],
                mean_confidence=float(np.mean(stats["confidences"])) if stats["confidences"] else None,
                ue_fma_exercise=stats["uefma"]["exercise_score"],
                ue_fma_estimate_after=mean_cal,
                ue_fma_ci95=None if sd_cal is None else 1.96 * sd_cal,
                ue_fma_estimate_after_raw=patient["ue_fma"]["mean"],
                ue_fma_update=stats["uefma"]["decision"],
                predicted_counts=stats["predicted_counts"],
                disagreements=stats["disagreements"],
                revisions=stats["revisions"],
                coordination_failures=stats["coordination_failures"],
                latency_ms={},
                clinician_flag=stats["clinician"]["clinician_flag"],
                clinician_reasons=stats["clinician"]["reasons"],
            )
            decision = svc.coordinator.exercise_decision(
                stats["status_counts"], exercise["n_windows"], bool(state.get("blocked_reason")), record.clinician_flag
            )
            rendered = svc.coordinator.render_report(*_report_fields(state, record, decision))
        record = record.model_copy(update={"latency_ms": dict(stats["latency_ms"])})
        agents = dict(patient["agents"])
        agents["escalation_rates"] = [*agents["escalation_rates"], stats["escalation_rate"]]
        svc.reports.append(rendered["text"])
        return {
            "patient": {**patient, "agents": agents, "exercises": [*patient["exercises"], record.model_dump()]},
            "exercise_report": {
                "text": rendered["text"],
                "details": rendered["details"],
                "decision": decision,
                "record": record.model_dump(),
            },
            "stats": stats,
        }

    def commit_memory(state: RehabState) -> dict:
        svc.perception.close(state["exercise"]["key"])
        svc.log.commit()
        assert_no_ground_truth(state["patient"], "patient")
        # Per-window scratch is dropped so the checkpoint holds only patient memory.
        return {"report": None, "proposals": {}, "verdict": None, "belief": None}

    builder = StateGraph(RehabState)
    for name, node in (
        ("start_exercise", start_exercise),
        ("observe", observe),
        ("propose", propose),
        ("safety_check", safety_check),
        ("coordinate", coordinate),
        ("deliver", deliver),
        ("assess", assess),
        ("exercise_review", exercise_review),
        ("report", report),
        ("commit_memory", commit_memory),
    ):
        builder.add_node(name, node)
    events = {"start": "start_exercise", "window": "observe", "end": "assess"}
    builder.add_conditional_edges(START, lambda s: events[s["event"]], list(events.values()))
    builder.add_edge("start_exercise", END)
    builder.add_edge("observe", "propose")
    builder.add_edge("propose", "safety_check")
    builder.add_edge("safety_check", "coordinate")
    builder.add_edge("coordinate", "deliver")
    builder.add_edge("deliver", END)
    builder.add_edge("assess", "exercise_review")
    builder.add_edge("exercise_review", "report")
    builder.add_edge("report", "commit_memory")
    builder.add_edge("commit_memory", END)
    return builder.compile(checkpointer=checkpointer)


def _format(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


_PRIMITIVE_WORDS = {
    "reach": "reaching",
    "transport": "carrying objects",
    "reposition": "repositioning",
    "stabilize": "holding steady",
    "rest": "resting",
}
_BAND_WORDS = {"severe": "severe", "moderate": "moderate", "mild": "mild"}


def _amount(count: int, total: int) -> str:
    share = count / total if total else 0.0
    return "a few" if share < 0.10 else "several" if share < 0.30 else "many"


def _movement_phrase(predicted_counts: dict[str, int]) -> str:
    total = sum(predicted_counts.values())
    if not total:
        return "could not be assessed (no usable sensor data)."
    active = sorted(
        ((count, name) for name, count in predicted_counts.items() if name != "rest" and count),
        reverse=True,
    )
    if not active:
        return "resting throughout; no active arm movement was seen."
    main = " and ".join(_PRIMITIVE_WORDS[name] for _, name in active[:2])
    rest_share = predicted_counts.get("rest", 0) / total
    rest = (
        "with little rest" if rest_share < 0.10
        else "with short rests between movements" if rest_share < 0.30
        else "with frequent rests"
    )
    return f"mostly {main}, {rest}."


def _trend_word(estimates: list[float], ci: float | None) -> str:
    known = [v for v in estimates if v is not None]
    if len(known) < 2:
        return "first assessment"
    tolerance = max(2.0, (ci or 0.0) / 2)
    change = known[-1] - known[0]
    if abs(change) <= tolerance:
        return "stable over recent exercises"
    return "improving over recent exercises" if change > 0 else "lower than in recent exercises"


def _report_fields(state: RehabState, record: ExerciseRecord, decision: str) -> tuple[dict, dict, dict]:
    """(clinician report fields, system details fields) for one exercise."""
    stats, patient, exercise = state["stats"], state["patient"], state["exercise"]
    counts = stats["status_counts"]
    windows = exercise["n_windows"]
    last = state.get("decision")
    last_released = last if last and last.get("released") else None
    previous = patient["exercises"][-(TREND_WINDOW - 1):] if TREND_WINDOW > 1 else []
    trend_uefma = [r["ue_fma_estimate_after"] for r in previous] + [record.ue_fma_estimate_after]
    trend_confidence = [r["mean_confidence"] for r in previous] + [record.mean_confidence]
    trend_escalation = patient["agents"]["escalation_rates"][-(TREND_WINDOW - 1):] + [stats["escalation_rate"]]
    clinician = stats["clinician"]
    band = patient["ue_fma"]["band"]

    # --- clinician report: plain language, patient-specific
    movement = _movement_phrase(stats["predicted_counts"])
    estimate, ci = record.ue_fma_estimate_after, record.ue_fma_ci95
    trend = _trend_word(trend_uefma, ci)
    if estimate is None:
        function = "not available (no usable sensor data)."
        function_words = "could not be assessed"
    else:
        low, high = max(0.0, estimate - (ci or 0.0)), min(66.0, estimate + (ci or 0.0))
        function = f"about {estimate:.0f} of 66 (likely {low:.0f}-{high:.0f}), {_BAND_WORDS.get(band, band)} impairment; {trend}."
        function_words = f"{_BAND_WORDS.get(band, band)} impairment, {trend}"
    if state.get("blocked_reason"):
        safety = "the exercise was stopped because the sensor data was unusable."
    else:
        notes = []
        if counts["revise"]:
            notes.append(
                f"{_amount(counts['revise'], windows)} movement prompts were switched to rest when the patient "
                "appeared to be resting"
            )
        if counts["escalate"]:
            notes.append(f"{_amount(counts['escalate'], windows)} prompts were held back because the movement was unclear")
        if counts["block"]:
            notes.append("some prompts were skipped because of unusable sensor data")
        safety = "no concerns." if not notes else "; ".join(notes).capitalize() + "."
    reasons = [plain_reason(reason) for reason in clinician["reasons"]]
    review = f"needed - {'; '.join(reasons)}." if clinician["clinician_flag"] else "not needed."
    if stats["usable_windows"] == 0:
        summary = "This exercise could not be assessed because the sensor data was unusable. Please review before the next exercise."
    else:
        summary = (
            f"The patient completed the {exercise['task_name'].replace('_', ' ')} task, {movement[:-1]}; "
            f"arm function shows {function_words}."
            + (" Please review before the next exercise." if clinician["clinician_flag"] else " No follow-up is needed.")
        )
    clinician_fields = {
        "subject_id": exercise["subject_id"],
        "exercise_number": exercise["exercise_index"] + 1,
        "task_label": exercise["task_name"].replace("_", " "),
        "movement": movement[0].upper() + movement[1:],
        "function": function,
        "safety": safety[0].upper() + safety[1:],
        "review": review[0].upper() + review[1:],
        "summary": summary,
    }

    # --- system details: the full technical record
    withheld = counts["escalate"] + counts["block"]
    safety_summary = (
        f"{withheld} windows withheld, {counts['revise']} revised"
        + (f"; blocked: {state['blocked_reason']}" if state.get("blocked_reason") else "")
    )
    details = {
        "subject_id": exercise["subject_id"],
        "exercise_number": exercise["exercise_index"] + 1,
        "clip_id": exercise["clip_id"],
        "decision": decision.upper(),
        "released": stats["released"],
        "windows": windows,
        "coverage": f"{stats['released'] / windows:.0%}" if windows else "0%",
        "approve": counts["approve"],
        "revise": counts["revise"],
        "escalate": counts["escalate"],
        "block": counts["block"],
        "next_cue": last_released["next_primitive"] if last_released else "none (withheld)",
        "next_confidence": _format(last_released["next_confidence"], 2) if last_released else "n/a",
        "plan": " -> ".join(last_released["final_action"]) if last_released else "withheld",
        "ue_exercise": _format(record.ue_fma_exercise),
        "ue_estimate": _format(record.ue_fma_estimate_after),
        "ue_ci": _format(record.ue_fma_ci95),
        "band": band or "n/a",
        "ue_update": record.ue_fma_update,
        "safety_summary": safety_summary,
        "disagreements": stats["disagreements"],
        "revisions": stats["revisions"],
        "failures": stats["coordination_failures"],
        "trend_window": len(trend_uefma),
        "trend_uefma": " -> ".join(_format(v) for v in trend_uefma),
        "trend_confidence": " -> ".join(_format(v, 2) for v in trend_confidence),
        "trend_escalation": " -> ".join(_format(v, 2) for v in trend_escalation),
        "clinician": "; ".join(clinician["reasons"]) if clinician["clinician_flag"] else "none",
    }
    return clinician_fields, details
