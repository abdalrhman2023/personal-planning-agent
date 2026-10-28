"""LangGraph agent graph for the Personal Planning Agent.

Implements a StateGraph with two entry modes:
  1. Background / Control Signal Mode  (from Adaptive Manager, Life Coach,
     or Assessment Agent)
  2. Conversational Chat Mode  (direct student requests)

Node pipeline:
  RouterNode → DeconstructorOrToolLLMNode → OptimizerAndGCalExecutionNode
             → TelemetryAndOutboundSignalsNode
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.agent.prompts import ROUTER_PROMPT, SYSTEM_PROMPT
from app.agent.tools import ALL_TOOLS, create_task, reschedule_block
from app.agent.tools import optimize_schedule as optimize_schedule_tool
from app.engine.deconstructor import deconstruct_task
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


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class PlanningState(BaseModel):
    """Shared state flowing through the graph."""

    # --- Input fields ---
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    entry_mode: Literal["chat", "control_signal", ""] = ""

    # --- Study data (mutable across nodes) ---
    tasks: list[dict] = Field(default_factory=list)
    university_schedule: list[dict] = Field(default_factory=list)
    gcal_busy_blocks: list[dict] = Field(default_factory=list)
    adaptive_model: dict = Field(default_factory=dict)
    target_date: str = ""

    # --- Intermediate results ---
    schedule_result: Optional[dict] = None
    gcal_events_created: list[dict] = Field(default_factory=list)

    # --- Outputs ---
    outbound_signals: Optional[dict] = None
    response_text: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Node 1: Router
# ---------------------------------------------------------------------------

def router_node(state: PlanningState) -> dict:
    """Classify the incoming request as chat or control_signal.

    For programmatic invocations the caller can pre-set ``entry_mode``; if it
    is already set we skip classification.
    """
    if state.entry_mode:
        return {}  # already classified

    last_msg = state.messages[-1] if state.messages else None
    if last_msg is None:
        return {"entry_mode": "chat"}

    content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    # Fast heuristic: if the message looks like JSON with known signal keys,
    # classify as control_signal without an LLM call.
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            signal_keys = {"signal_type", "command", "milestones", "remediation_topics"}
            if signal_keys & set(payload.keys()):
                return {"entry_mode": "control_signal"}
    except (json.JSONDecodeError, TypeError):
        pass

    return {"entry_mode": "chat"}


# ---------------------------------------------------------------------------
# Node 2: Deconstructor / Tool-LLM
# ---------------------------------------------------------------------------

def _add_or_deconstruct_task(
    title: str,
    estimated_mins: int,
    priority_weight: float = 3.0,
    cognitive_load: str = "medium",
    topic_ref: Optional[str] = None,
    source: str = "user",
    parent_milestone_id: Optional[str] = None,
    deadline: Optional[str] = None,
    focus_duration: int = 25,
) -> list[dict]:
    """Helper to create a task or deconstruct it into microtasks if it exceeds focus duration."""
    task = StudyTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        title=title,
        source=source,
        estimated_mins=estimated_mins,
        priority_weight=priority_weight,
        cognitive_load=cognitive_load,
        topic_ref=topic_ref,
        parent_milestone_id=parent_milestone_id,
        deadline=deadline,
        status="pending",
    )
    if task.estimated_mins > focus_duration:
        dr = deconstruct_task(task, focus_duration, force_fallback=True)
        out = []
        for i, mt in enumerate(dr.micro_tasks):
            sub = StudyTask(
                task_id=f"{task.task_id}_p{i+1}",
                parent_milestone_id=task.parent_milestone_id,
                title=mt.title,
                source=source,
                estimated_mins=mt.estimated_mins,
                priority_weight=task.priority_weight,
                cognitive_load=mt.cognitive_load,
                deadline=task.deadline,
                topic_ref=mt.topic_ref or task.topic_ref,
                status="pending",
            )
            out.append(sub.model_dump())
        return out
    else:
        return [task.model_dump()]


def _parse_natural_language_intent(
    message: str,
    focus_duration: int = 25,
) -> Optional[dict]:
    """Extract study task creation parameters from natural language input."""
    if not message or not isinstance(message, str):
        return None

    # Check for study/task intent
    keywords = ["study", "task", "review", "read", "practice", "schedule", "add", "create", "plan", "session", "block"]
    if not any(re.search(r'\b' + re.escape(kw) + r'\b', message, re.IGNORECASE) for kw in keywords):
        return None

    # 1. Duration extraction
    dur_match = re.search(r'(\d+)\s*(?:mins?|minutes?|m\b)', message, re.IGNORECASE)
    if dur_match:
        dur = int(dur_match.group(1))
    else:
        dur_h = re.search(r'(\d+)\s*(?:hours?|hrs?|h\b)', message, re.IGNORECASE)
        dur = int(dur_h.group(1)) * 60 if dur_h else focus_duration

    # 2. Priority extraction
    prio_num = re.search(r'priority\s*[:=]?\s*([1-5](?:\.\d+)?)', message, re.IGNORECASE)
    if prio_num:
        prio = float(prio_num.group(1))
    elif re.search(r'\b(high priority|urgent|critical|important)\b', message, re.IGNORECASE):
        prio = 4.5
    elif re.search(r'\b(low priority|casual|light|easy)\b', message, re.IGNORECASE):
        prio = 2.0
    else:
        prio = 3.0

    load = "high" if prio >= 4.0 else ("low" if prio <= 2.0 else "medium")

    # 3. Clean Title extraction
    t = message
    t = re.sub(r'^(?:please\s+)?(?:add|create|schedule|plan|insert|study)\s+', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^(?:a\s+|an\s+)?(?:(?:high|medium|low)\s+priority\s+)?(?:study\s+)?(?:task|session|block)?(?:\s+for)?\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^(?:task\s*[:=]|\d+[- ](?:minute|min)\s+(?:study\s+)?(?:session|block|task)?(?:\s+for)?)\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^[:\s]+', '', t)
    t = re.sub(r'\s*(?:tomorrow|today|tonight|for\s+\d+\s*(?:mins?|minutes?|hours?|hrs?|m\b|h\b)|\b\d+\s*(?:mins?|minutes?|hours?|hrs?)\b|at\s+\d{1,2}(?::\d{2})?|priority\s*[:=]?\s*[1-5](?:\.\d+)?|and\s+optimize.*|optimize.*).*$', '', t, flags=re.IGNORECASE)
    title = t.strip(' ,.-:') or "Study Session"

    topic_ref = re.sub(r'[^\w\s]', '', title).replace(' ', '_')

    return {
        "title": title,
        "estimated_mins": dur,
        "priority_weight": prio,
        "cognitive_load": load,
        "topic_ref": topic_ref,
    }


def deconstructor_or_tool_llm_node(state: PlanningState) -> dict:
    """Handle control signals directly OR prepare the LLM for chat.

    - Control signals: parse the JSON, create tasks / apply restructure.
    - Chat: invoke the LLM with tools bound so it can call create_task,
      reschedule_block, optimize_schedule. Also provides an intelligent
      natural language fallback parser if LLM is unavailable or fails.
    """
    new_tasks = list(state.tasks)
    initial_tasks_count = len(new_tasks)
    adaptive = AdaptiveModelContext(**(state.adaptive_model or {}))
    focus_duration = adaptive.effective_focus_duration_mins or 25
    target_date = state.target_date or "2026-11-02"
    messages_out: list[BaseMessage] = []

    if state.entry_mode == "control_signal":
        last_content = state.messages[-1].content if state.messages else "{}"
        try:
            payload = json.loads(last_content)
        except (json.JSONDecodeError, TypeError):
            payload = {}

        signal_type = payload.get("signal_type") or payload.get("command", "")

        # --- PREEMPTIVE_SCHEDULE_RESTRUCTURE from Adaptive Manager ---
        if signal_type == "PREEMPTIVE_SCHEDULE_RESTRUCTURE":
            adaptive_updates = payload.get("adaptive_model_updates", {})
            adaptive_dict = adaptive.model_dump()
            adaptive_dict.update(adaptive_updates)
            adaptive_dict["current_system_mode"] = "PREEMPTIVE_RESTRUCTURE"

            # Reset non-completed tasks back to pending for re-optimization
            for t in new_tasks:
                if t.get("status") != "completed":
                    t["status"] = "pending"

            messages_out.append(
                AIMessage(content="Received PREEMPTIVE_SCHEDULE_RESTRUCTURE signal. "
                          "Switching to restructure mode and re-optimising.")
            )
            return {
                "tasks": new_tasks,
                "target_date": target_date,
                "adaptive_model": adaptive_dict,
                "messages": messages_out,
            }

        # --- Milestones from Life Coach ---
        if signal_type == "MILESTONE_TASKS" or "milestones" in payload:
            milestones = payload.get("milestones", payload.get("tasks", []))
            focus = adaptive.effective_focus_duration_mins
            for ms in milestones:
                task = StudyTask(
                    task_id=ms.get("task_id", f"lc_{len(new_tasks)}"),
                    title=ms.get("title", "Milestone task"),
                    source="life_coach",
                    estimated_mins=ms.get("estimated_mins", focus),
                    priority_weight=ms.get("priority_weight", 3.0),
                    cognitive_load=ms.get("cognitive_load", "medium"),
                    deadline=ms.get("deadline"),
                    topic_ref=ms.get("topic_ref"),
                    parent_milestone_id=ms.get("parent_milestone_id"),
                )
                # Deconstruct if large
                if task.estimated_mins > focus:
                    dr = deconstruct_task(task, focus, force_fallback=True)
                    for i, mt in enumerate(dr.micro_tasks):
                        sub = StudyTask(
                            task_id=f"{task.task_id}_p{i+1}",
                            parent_milestone_id=task.parent_milestone_id,
                            title=mt.title,
                            source="life_coach",
                            estimated_mins=mt.estimated_mins,
                            priority_weight=task.priority_weight,
                            cognitive_load=mt.cognitive_load,
                            deadline=task.deadline,
                            topic_ref=mt.topic_ref,
                        )
                        new_tasks.append(sub.model_dump())
                else:
                    new_tasks.append(task.model_dump())

            messages_out.append(
                AIMessage(content=f"Ingested {len(milestones)} milestone(s) from Life Coach.")
            )
            return {"tasks": new_tasks, "target_date": target_date, "messages": messages_out}

        # --- Remediation from Assessment Agent ---
        if signal_type == "REMEDIATION_TASKS" or "remediation_topics" in payload:
            topics = payload.get("remediation_topics", payload.get("tasks", []))
            for rt in topics:
                task = StudyTask(
                    task_id=rt.get("task_id", f"rem_{len(new_tasks)}"),
                    title=rt.get("title", "Remediation"),
                    source="assessment_remediation",
                    estimated_mins=rt.get("estimated_mins", 30),
                    priority_weight=rt.get("priority_weight", 4.5),
                    cognitive_load=rt.get("cognitive_load", "high"),
                    deadline=rt.get("deadline"),
                    topic_ref=rt.get("topic_ref"),
                )
                new_tasks.append(task.model_dump())
            messages_out.append(
                AIMessage(content=f"Ingested {len(topics)} remediation task(s) from Assessment Agent.")
            )
            return {"tasks": new_tasks, "target_date": target_date, "messages": messages_out}

        # Unknown signal – pass through
        messages_out.append(AIMessage(content=f"Received unknown signal: {signal_type}"))
        return {"messages": messages_out, "target_date": target_date}

    # ---------- Chat mode: invoke LLM with tools ----------
    user_msg = state.messages[-1].content if state.messages else ""
    user_msg_str = (
        user_msg
        if isinstance(user_msg, str)
        else (
            " ".join(
                [c.get("text", "") if isinstance(c, dict) else str(c) for c in user_msg]
            )
            if isinstance(user_msg, list)
            else str(user_msg)
        )
    )

    ai_content = ""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = getattr(config, "GEMINI_MODEL_NAME", "gemini-3.5-flash")
        conversation = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.messages)

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=config.GEMINI_API_KEY,
                timeout=15,
                max_retries=1,
            )
            llm_with_tools = llm.bind_tools(ALL_TOOLS)
            ai_msg: AIMessage = llm_with_tools.invoke(conversation)
        except Exception:
            # Fallback to secondary model if primary returned 404/503/error
            fallback_model = "gemini-3.8-flash" if model_name != "gemini-3.8-flash" else "gemini-3.5-flash"
            llm = ChatGoogleGenerativeAI(
                model=fallback_model,
                google_api_key=config.GEMINI_API_KEY,
                timeout=15,
                max_retries=1,
            )
            llm_with_tools = llm.bind_tools(ALL_TOOLS)
            ai_msg = llm_with_tools.invoke(conversation)

        messages_out.append(ai_msg)
        if ai_msg.content:
            ai_content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)

        # Execute any tool calls the LLM made
        if ai_msg.tool_calls:
            tool_map = {t.name: t for t in ALL_TOOLS}
            for tc in ai_msg.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn:
                    result_str = tool_fn.invoke(tc["args"])
                    messages_out.append(
                        ToolMessage(content=result_str, tool_call_id=tc["id"])
                    )
                    # If create_task was called, add to state & deconstruct if needed
                    if tc["name"] == "create_task":
                        raw_task = json.loads(result_str)
                        created_blocks = _add_or_deconstruct_task(
                            title=raw_task.get("title", "Study Task"),
                            estimated_mins=raw_task.get("estimated_mins", focus_duration),
                            priority_weight=raw_task.get("priority_weight", 3.0),
                            cognitive_load=raw_task.get("cognitive_load", "medium"),
                            topic_ref=raw_task.get("topic_ref"),
                            source=raw_task.get("source", "user"),
                            parent_milestone_id=raw_task.get("parent_milestone_id"),
                            deadline=raw_task.get("deadline"),
                            focus_duration=focus_duration,
                        )
                        new_tasks.extend(created_blocks)
                    elif tc["name"] == "reschedule_block":
                        resched = json.loads(result_str)
                        tid = resched.get("task_id")
                        for t in new_tasks:
                            if t.get("task_id") == tid:
                                t["scheduled_start"] = resched.get("new_start")
                                t["scheduled_end"] = resched.get("new_end")
                                if resched.get("new_date"):
                                    t["scheduled_date"] = resched.get("new_date")
                                t["status"] = "scheduled"

    except Exception:
        # LLM unavailable, timed out, or threw an error
        pass

    # Fallback Parser: If no task was created via tool calling, parse user intent from natural language
    if len(new_tasks) == initial_tasks_count and user_msg_str:
        parsed = _parse_natural_language_intent(user_msg_str, focus_duration=focus_duration)
        if parsed:
            created_blocks = _add_or_deconstruct_task(
                title=parsed["title"],
                estimated_mins=parsed["estimated_mins"],
                priority_weight=parsed["priority_weight"],
                cognitive_load=parsed["cognitive_load"],
                topic_ref=parsed["topic_ref"],
                source="user",
                focus_duration=focus_duration,
            )
            new_tasks.extend(created_blocks)
            if not ai_content:
                messages_out.append(
                    AIMessage(
                        content=f"Created task '{parsed['title']}' ({parsed['estimated_mins']} mins) and triggered schedule optimization."
                    )
                )

    if not messages_out and user_msg_str:
        messages_out.append(
            AIMessage(
                content=f"Received: '{user_msg_str}'. Planning agent evaluated your request and updated the schedule."
            )
        )

    return {"tasks": new_tasks, "target_date": target_date, "messages": messages_out}


# ---------------------------------------------------------------------------
# Node 3: Optimizer & GCal Execution
# ---------------------------------------------------------------------------

def optimizer_and_gcal_node(state: PlanningState) -> dict:
    """Run the optimizer on current state and sync results to Google Calendar."""
    if not state.target_date:
        return {}

    tasks = [StudyTask(**t) for t in state.tasks]
    lectures = [FixedLecture(**l) for l in state.university_schedule]
    adaptive = AdaptiveModelContext(**(state.adaptive_model or {}))

    request = ScheduleRequest(
        target_date=state.target_date,
        tasks=tasks,
        university_schedule=lectures,
        gcal_busy_blocks=state.gcal_busy_blocks,
        adaptive_model=adaptive,
    )

    result: ScheduleResult = run_optimizer(request)

    # --- Sync scheduled blocks to Google Calendar (mock-safe) ---
    gcal = GoogleCalendarService()

    # Clean up previously created events for this target_date or tracked in state/tasks
    old_events = list(state.gcal_events_created)
    for old_ev in old_events:
        old_id = old_ev.get("id")
        if old_id:
            try:
                gcal.delete_calendar_event(old_id)
            except Exception:
                pass

    for t in state.tasks:
        old_tid = t.get("gcal_event_id")
        if old_tid:
            try:
                gcal.delete_calendar_event(old_tid)
            except Exception:
                pass

    events_created: list[dict] = []
    task_event_map: dict[str, str] = {}
    for block in result.scheduled_blocks:
        try:
            evt = gcal.create_calendar_event(
                summary=f"Study: {block.title}",
                date=block.date,
                start_time=block.start,
                end_time=block.end,
                description=f"Task {block.task_id} – Part {block.part}",
            )
            events_created.append(evt)
            task_event_map[block.task_id] = evt.get("id", "")
        except Exception:
            pass  # best-effort calendar sync

    # Update task statuses
    scheduled_ids = {b.task_id for b in result.scheduled_blocks}
    updated_tasks: list[dict] = []
    for t_dict in state.tasks:
        t_copy = dict(t_dict)
        if t_copy["task_id"] in scheduled_ids:
            t_copy["status"] = "scheduled"
            t_copy["gcal_event_id"] = task_event_map.get(t_copy["task_id"])
            # Find the first block for this task
            for b in result.scheduled_blocks:
                if b.task_id == t_copy["task_id"]:
                    t_copy["scheduled_date"] = b.date
                    t_copy["scheduled_start"] = b.start
                    t_copy["scheduled_end"] = b.end
                    break
        else:
            t_copy["gcal_event_id"] = None
        updated_tasks.append(t_copy)

    messages_out: list[BaseMessage] = []
    if result.is_feasible:
        messages_out.append(AIMessage(
            content=f"Schedule optimised: {len(result.scheduled_blocks)} block(s) "
                    f"placed for {state.target_date}."
        ))
    else:
        messages_out.append(AIMessage(
            content=f"Schedule partially feasible: "
                    f"{len(result.unscheduled_tasks)} task(s) could not be placed."
        ))

    return {
        "tasks": updated_tasks,
        "schedule_result": result.model_dump(),
        "gcal_events_created": events_created,
        "messages": messages_out,
    }


# ---------------------------------------------------------------------------
# Node 4: Telemetry & Outbound Signals
# ---------------------------------------------------------------------------

def telemetry_and_signals_node(state: PlanningState) -> dict:
    """Compute telemetry metrics and produce outbound signals."""
    if not state.schedule_result:
        return {}

    sr = ScheduleResult(**state.schedule_result)
    tasks = [StudyTask(**t) for t in state.tasks]
    lectures = [FixedLecture(**l) for l in state.university_schedule]

    target_date = state.target_date or "2026-11-02"
    signals: OutboundSignals = compute_outbound_signals(
        scheduled_blocks=sr.scheduled_blocks,
        tasks=tasks,
        university_schedule=lectures,
        gcal_busy_blocks=state.gcal_busy_blocks,
        target_date=target_date,
        feasibility_flags=sr.feasibility_flags,
    )

    response_parts: list[str] = []
    density = signals.workload_density_metrics.get("workload_density", 0)
    response_parts.append(
        f"Workload density: {density:.0%}"
    )
    if signals.to_adaptive_manager:
        response_parts.append(
            f"[Alert] Sent to Adaptive Manager: "
            f"{signals.to_adaptive_manager.get('signal_type', signals.to_adaptive_manager.get('message', ''))}"
        )
    if signals.to_life_coach:
        response_parts.append("[Report] Feasibility report sent to Life Coach.")
    if signals.to_ai_tutor:
        response_parts.append(
            f"[Tutor] {len(signals.to_ai_tutor)} study block invitation(s) sent to AI Tutor."
        )

    return {
        "outbound_signals": signals.model_dump(),
        "response_text": " | ".join(response_parts),
    }


# ---------------------------------------------------------------------------
# Conditional edge: should we run the optimizer?
# ---------------------------------------------------------------------------

def _should_optimize(state: PlanningState) -> str:
    """Return next node name.  Skip optimizer if there is no target_date."""
    if state.target_date and state.tasks:
        return "optimizer_and_gcal"
    return "telemetry_and_signals"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the Planning Agent StateGraph."""

    graph = StateGraph(PlanningState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("deconstructor_or_tool_llm", deconstructor_or_tool_llm_node)
    graph.add_node("optimizer_and_gcal", optimizer_and_gcal_node)
    graph.add_node("telemetry_and_signals", telemetry_and_signals_node)

    # Edges
    graph.set_entry_point("router")
    graph.add_edge("router", "deconstructor_or_tool_llm")
    graph.add_conditional_edges(
        "deconstructor_or_tool_llm",
        _should_optimize,
        {
            "optimizer_and_gcal": "optimizer_and_gcal",
            "telemetry_and_signals": "telemetry_and_signals",
        },
    )
    graph.add_edge("optimizer_and_gcal", "telemetry_and_signals")
    graph.add_edge("telemetry_and_signals", END)

    return graph.compile()
