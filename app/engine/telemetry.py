"""Telemetry Engine – computes workload metrics, velocity slip, and study-block invitations.

All computations are pure Python / deterministic.  No LLM calls.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    OutboundSignals,
    ScheduledBlock,
    StudyTask,
)
from app.engine.optimizer import _hhmm_to_mins, _DAY_START, _DAY_END


# ---------------------------------------------------------------------------
# 1. Workload Density Metrics
# ---------------------------------------------------------------------------

def compute_workload_density(
    scheduled_blocks: List[ScheduledBlock],
    university_schedule: List[FixedLecture],
    gcal_busy_blocks: List[Dict[str, str]],
    target_date: str,
) -> Dict[str, float]:
    """Compute workload density = scheduled_study_mins / total_free_mins.

    *total_free_mins* is the operating window (08:00-22:00) minus fixed
    lectures and external calendar blocks on the target date.
    """
    total_operating_mins = _DAY_END - _DAY_START  # 840

    # Subtract fixed lectures on target_date
    lecture_mins = sum(
        _hhmm_to_mins(lec.end) - _hhmm_to_mins(lec.start)
        for lec in university_schedule
        if lec.date == target_date
    )

    # Subtract gcal busy blocks
    gcal_mins = sum(
        _hhmm_to_mins(blk["end"]) - _hhmm_to_mins(blk["start"])
        for blk in gcal_busy_blocks
    )

    total_free_mins = max(1, total_operating_mins - lecture_mins - gcal_mins)

    scheduled_study_mins = sum(
        _hhmm_to_mins(b.end) - _hhmm_to_mins(b.start)
        for b in scheduled_blocks
        if b.date == target_date
    )

    density = round(scheduled_study_mins / total_free_mins, 4)

    return {
        "target_date": target_date,
        "total_operating_mins": total_operating_mins,
        "lecture_mins": lecture_mins,
        "gcal_busy_mins": gcal_mins,
        "total_free_mins": total_free_mins,
        "scheduled_study_mins": scheduled_study_mins,
        "workload_density": density,
    }


# ---------------------------------------------------------------------------
# 2. Velocity Slip Detection
# ---------------------------------------------------------------------------

_VELOCITY_SLIP_THRESHOLD = 0.20  # 20%


def compute_velocity_slip(
    tasks: List[StudyTask],
) -> Dict:
    """Compare completed tasks vs scheduled+missed tasks.

    Returns a dict with ``slip_ratio``, ``deviation_exceeded`` flag, and,
    when the threshold is crossed, an ``alert_payload`` for the Adaptive
    Manager.

    Only tasks that have *entered the schedule* (status in scheduled,
    completed, missed) are considered.  Pending/delayed tasks are excluded
    because they haven't been attempted yet.
    """
    completed = [t for t in tasks if t.status == "completed"]
    scheduled_or_missed = [
        t for t in tasks if t.status in ("scheduled", "completed", "missed")
    ]

    total_attempted = len(scheduled_or_missed)
    if total_attempted == 0:
        return {
            "completed_count": 0,
            "attempted_count": 0,
            "completion_rate": 1.0,
            "expected_rate": 1.0,
            "slip_ratio": 0.0,
            "deviation_exceeded": False,
            "alert_payload": None,
        }

    completion_rate = len(completed) / total_attempted
    expected_rate = 1.0  # we expect all scheduled tasks to be completed
    slip_ratio = round(abs(expected_rate - completion_rate), 4)
    deviation_exceeded = slip_ratio > _VELOCITY_SLIP_THRESHOLD

    alert_payload = None
    if deviation_exceeded:
        alert_payload = {
            "signal_type": "VELOCITY_SLIP_ALERT",
            "completed_count": len(completed),
            "attempted_count": total_attempted,
            "completion_rate": round(completion_rate, 4),
            "slip_ratio": slip_ratio,
            "message": (
                f"Velocity slip detected: completion rate "
                f"{completion_rate:.0%} deviates from expected by "
                f"{slip_ratio:.0%} (threshold {_VELOCITY_SLIP_THRESHOLD:.0%})."
            ),
        }

    return {
        "completed_count": len(completed),
        "attempted_count": total_attempted,
        "completion_rate": round(completion_rate, 4),
        "expected_rate": expected_rate,
        "slip_ratio": slip_ratio,
        "deviation_exceeded": deviation_exceeded,
        "alert_payload": alert_payload,
    }


# ---------------------------------------------------------------------------
# 3. Study Block Invitations for AI Tutor Agent
# ---------------------------------------------------------------------------

def generate_study_block_invitations(
    scheduled_blocks: List[ScheduledBlock],
    tasks: List[StudyTask],
) -> List[dict]:
    """Generate Study Block Invitations & Context Payload for AI Tutor Agent.

    Each invitation carries enough context for the Tutor to prepare the
    right material when the study session starts.
    """
    task_lookup: Dict[str, StudyTask] = {t.task_id: t for t in tasks}
    invitations: List[dict] = []

    for block in scheduled_blocks:
        task = task_lookup.get(block.task_id)
        if task is None:
            continue

        invitations.append({
            "block_date": block.date,
            "block_start": block.start,
            "block_end": block.end,
            "block_part": block.part,
            "task_id": task.task_id,
            "task_title": task.title,
            "topic_ref": task.topic_ref,
            "source": task.source,
            "cognitive_load": task.cognitive_load,
            "parent_milestone_id": task.parent_milestone_id,
        })

    return invitations


# ---------------------------------------------------------------------------
# 4. Aggregate – produce full OutboundSignals
# ---------------------------------------------------------------------------

def compute_outbound_signals(
    scheduled_blocks: List[ScheduledBlock],
    tasks: List[StudyTask],
    university_schedule: List[FixedLecture],
    gcal_busy_blocks: List[Dict[str, str]],
    target_date: str,
    feasibility_flags: Optional[dict] = None,
) -> OutboundSignals:
    """One-call convenience that assembles all outbound signals."""

    # Workload density
    density = compute_workload_density(
        scheduled_blocks, university_schedule, gcal_busy_blocks, target_date,
    )

    # Velocity slip
    slip = compute_velocity_slip(tasks)

    # Adaptive Manager signal: feasibility flags take priority, then slip alert
    to_adaptive: Optional[dict] = None
    if feasibility_flags:
        to_adaptive = feasibility_flags
    elif slip["deviation_exceeded"]:
        to_adaptive = slip["alert_payload"]

    # Life Coach: send feasibility report if schedule is infeasible
    to_life_coach: Optional[dict] = None
    if feasibility_flags:
        to_life_coach = {
            "signal_type": "MILESTONE_FEASIBILITY_REPORT",
            "details": feasibility_flags,
        }

    # AI Tutor invitations
    to_tutor = generate_study_block_invitations(scheduled_blocks, tasks)

    return OutboundSignals(
        to_adaptive_manager=to_adaptive,
        to_life_coach=to_life_coach,
        to_ai_tutor=to_tutor,
        workload_density_metrics=density,
    )
