"""FastAPI HTTP Routes for the Personal Planning Agent (Agent 3).

Exposes the following endpoints:
  - POST /api/v1/planning/optimize   – Run deterministic optimizer + calendar sync
  - POST /api/v1/planning/chat       – Interactive LangGraph agent conversation / control signal
  - POST /api/v1/planning/telemetry  – Compute workload density, velocity slip, and tutor signals
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from app.agent.graph import PlanningState, build_graph
from app.engine.optimizer import optimize_schedule as run_optimizer
from app.engine.telemetry import compute_outbound_signals
from app.integrations.gcal_service import GoogleCalendarService
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    OutboundSignals,
    ScheduleRequest,
    ScheduleResult,
    ScheduledBlock,
    StudyTask,
)

router = APIRouter(prefix="/api/v1/planning", tags=["planning"])


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class OptimizeResponse(BaseModel):
    is_feasible: bool
    scheduled_blocks: List[ScheduledBlock] = Field(default_factory=list)
    unscheduled_tasks: List[str] = Field(default_factory=list)
    feasibility_flags: Optional[dict] = None
    schedule_result: Optional[ScheduleResult] = None
    outbound_signals: Optional[OutboundSignals] = None
    gcal_events: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = "Please schedule a 25-minute study block for Algorithms today"
    target_date: Optional[str] = "2026-11-02"
    tasks: List[StudyTask] = Field(default_factory=list)
    university_schedule: List[FixedLecture] = Field(default_factory=list)
    gcal_busy_blocks: List[Dict[str, str]] = Field(default_factory=list)
    adaptive_model: AdaptiveModelContext = Field(default_factory=AdaptiveModelContext)
    entry_mode: Optional[str] = None  # "chat" or "control_signal"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Move my evening study session to 19:30",
                "target_date": "2026-11-02",
                "tasks": [
                    {
                        "task_id": "task_algo",
                        "title": "Algorithms Practice",
                        "source": "user",
                        "estimated_mins": 25,
                        "priority_weight": 3.5,
                        "cognitive_load": "medium",
                        "status": "pending"
                    }
                ],
                "university_schedule": [],
                "gcal_busy_blocks": [],
                "adaptive_model": {
                    "peak_productivity_window": "09:00-11:00",
                    "secondary_productivity_window": "19:00-22:00",
                    "effective_focus_duration_mins": 25,
                    "max_daily_study_mins": 240,
                    "current_system_mode": "NORMAL"
                },
                "entry_mode": "chat"
            }
        }
    )


class ChatResponse(BaseModel):
    response: str
    entry_mode: str
    tasks: List[StudyTask] = Field(default_factory=list)
    schedule_result: Optional[ScheduleResult] = None
    outbound_signals: Optional[OutboundSignals] = None
    gcal_events: List[Dict[str, Any]] = Field(default_factory=list)


class TelemetryRequest(BaseModel):
    target_date: str = "2026-11-02"
    scheduled_blocks: List[ScheduledBlock] = Field(default_factory=list)
    tasks: List[StudyTask] = Field(default_factory=list)
    university_schedule: List[FixedLecture] = Field(default_factory=list)
    gcal_busy_blocks: List[Dict[str, str]] = Field(default_factory=list)
    feasibility_flags: Optional[dict] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_date": "2026-11-02",
                "scheduled_blocks": [
                    {
                        "task_id": "task_1",
                        "title": "Calculus Study Block",
                        "date": "2026-11-02",
                        "start": "11:00",
                        "end": "11:25",
                        "part": 1
                    }
                ],
                "tasks": [
                    {
                        "task_id": "task_1",
                        "title": "Calculus Study Block",
                        "source": "user",
                        "estimated_mins": 25,
                        "priority_weight": 4.0,
                        "cognitive_load": "high",
                        "status": "completed"
                    }
                ],
                "university_schedule": [],
                "gcal_busy_blocks": []
            }
        }
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/optimize", response_model=OptimizeResponse)
def optimize_schedule_endpoint(request: ScheduleRequest) -> OptimizeResponse:
    """Run the deterministic scheduler for target_date and sync to Google Calendar."""
    # 1. Deterministic Optimization
    result = run_optimizer(request)

    # 2. Calendar Event Creation (Safe in both Mock and Live mode)
    gcal = GoogleCalendarService()

    # Clean up previous gcal_event_ids present on tasks to prevent duplicates
    for task in request.tasks:
        if task.gcal_event_id:
            try:
                gcal.delete_calendar_event(task.gcal_event_id)
            except Exception:
                pass

    gcal_events: List[Dict[str, Any]] = []
    for block in result.scheduled_blocks:
        try:
            event = gcal.create_calendar_event(
                summary=f"Study: {block.title}",
                date=block.date,
                start_time=block.start,
                end_time=block.end,
                description=f"Task ID: {block.task_id} (Part {block.part})",
            )
            gcal_events.append(event)
        except Exception:
            pass

    # 3. Compute Outbound Signals
    signals = compute_outbound_signals(
        scheduled_blocks=result.scheduled_blocks,
        tasks=request.tasks,
        university_schedule=request.university_schedule,
        gcal_busy_blocks=request.gcal_busy_blocks,
        target_date=request.target_date,
        feasibility_flags=result.feasibility_flags,
    )

    return OptimizeResponse(
        is_feasible=result.is_feasible,
        scheduled_blocks=result.scheduled_blocks,
        unscheduled_tasks=result.unscheduled_tasks,
        feasibility_flags=result.feasibility_flags,
        schedule_result=result,
        outbound_signals=signals,
        gcal_events=gcal_events,
    )


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle conversational user requests or multi-agent control signals via LangGraph."""
    try:
        graph = build_graph()
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "entry_mode": request.entry_mode or "",
            "target_date": request.target_date or "",
            "tasks": [t.model_dump() for t in request.tasks],
            "university_schedule": [l.model_dump() for l in request.university_schedule],
            "gcal_busy_blocks": request.gcal_busy_blocks,
            "adaptive_model": request.adaptive_model.model_dump(),
        }

        final_state = graph.invoke(initial_state)

        # Extract reply message
        response_text = final_state.get("response_text", "")
        messages = final_state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content") and last_msg.content:
                msg_content = str(last_msg.content)
                if response_text:
                    response_text = f"{msg_content}\n\n[Telemetry] {response_text}"
                else:
                    response_text = msg_content

        updated_tasks = [StudyTask(**t) for t in final_state.get("tasks", [])]
        raw_sr = final_state.get("schedule_result")
        schedule_result = ScheduleResult(**raw_sr) if raw_sr else None

        raw_signals = final_state.get("outbound_signals")
        outbound_signals = OutboundSignals(**raw_signals) if raw_signals else None

        return ChatResponse(
            response=response_text or "Processed successfully.",
            entry_mode=final_state.get("entry_mode", "chat"),
            tasks=updated_tasks,
            schedule_result=schedule_result,
            outbound_signals=outbound_signals,
            gcal_events=final_state.get("gcal_events_created", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry", response_model=OutboundSignals)
def telemetry_endpoint(request: TelemetryRequest) -> OutboundSignals:
    """Generate telemetry metrics, velocity slip analysis, and study block invitations."""
    signals = compute_outbound_signals(
        scheduled_blocks=request.scheduled_blocks,
        tasks=request.tasks,
        university_schedule=request.university_schedule,
        gcal_busy_blocks=request.gcal_busy_blocks,
        target_date=request.target_date,
        feasibility_flags=request.feasibility_flags,
    )
    return signals
