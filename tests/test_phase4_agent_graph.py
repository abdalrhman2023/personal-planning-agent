"""Phase 4 tests – LangGraph Agent & Tool-Driven Execution.

Tests cover:
  A. Tool unit tests (create_task, reschedule_block, optimize_schedule)
  B. Prompt constants
  C. Graph state schema
  D. Router node
  E. Control signal mode – PREEMPTIVE_SCHEDULE_RESTRUCTURE
  F. Control signal mode – Life Coach milestones
  G. Control signal mode – Assessment Agent remediation
  H. Optimizer & GCal execution node
  I. Telemetry & outbound signals node
  J. Full graph integration (end-to-end control signal flow)
  K. Full graph integration (chat mode fallback)
"""

import json
import pytest

from langchain_core.messages import AIMessage, HumanMessage

from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    ScheduleResult,
    StudyTask,
)
from app.agent.tools import (
    ALL_TOOLS,
    create_task,
    reschedule_block,
    optimize_schedule,
)
from app.agent.prompts import ROUTER_PROMPT, SYSTEM_PROMPT
from app.agent.graph import (
    PlanningState,
    build_graph,
    deconstructor_or_tool_llm_node,
    optimizer_and_gcal_node,
    router_node,
    telemetry_and_signals_node,
)


# ===================================================================
# Helpers
# ===================================================================

def _task_dict(**overrides) -> dict:
    defaults = dict(
        task_id="t1",
        title="Review Ch. 1",
        source="user",
        estimated_mins=25,
        priority_weight=3.0,
        cognitive_load="medium",
        status="pending",
    )
    defaults.update(overrides)
    return defaults


def _state(**overrides) -> PlanningState:
    defaults = dict(
        target_date="2026-11-02",
        tasks=[_task_dict()],
        adaptive_model=AdaptiveModelContext().model_dump(),
    )
    defaults.update(overrides)
    return PlanningState(**defaults)


# ===================================================================
# A. Tool unit tests
# ===================================================================

class TestCreateTaskTool:
    def test_returns_valid_json(self):
        result = create_task.invoke({
            "title": "Read chapter 5",
            "estimated_mins": 45,
        })
        data = json.loads(result)
        assert data["title"] == "Read chapter 5"
        assert data["estimated_mins"] == 45
        assert data["task_id"].startswith("task_")

    def test_default_fields(self):
        result = create_task.invoke({
            "title": "X",
            "estimated_mins": 10,
        })
        data = json.loads(result)
        assert data["source"] == "user"
        assert data["cognitive_load"] == "medium"
        assert data["priority_weight"] == 3.0
        assert data["status"] == "pending"

    def test_custom_fields(self):
        result = create_task.invoke({
            "title": "Fix gaps",
            "estimated_mins": 30,
            "source": "assessment_remediation",
            "cognitive_load": "high",
            "priority_weight": 4.5,
            "deadline": "2026-11-03T23:59:59",
            "topic_ref": "probability",
        })
        data = json.loads(result)
        assert data["source"] == "assessment_remediation"
        assert data["cognitive_load"] == "high"
        assert data["topic_ref"] == "probability"


class TestRescheduleBlockTool:
    def test_returns_valid_json(self):
        result = reschedule_block.invoke({
            "task_id": "t1",
            "new_start": "14:00",
            "new_end": "14:25",
        })
        data = json.loads(result)
        assert data["action"] == "reschedule_block"
        assert data["task_id"] == "t1"
        assert data["new_start"] == "14:00"

    def test_optional_new_date(self):
        result = reschedule_block.invoke({
            "task_id": "t1",
            "new_start": "14:00",
            "new_end": "14:25",
            "new_date": "2026-11-03",
        })
        data = json.loads(result)
        assert data["new_date"] == "2026-11-03"


class TestOptimizeScheduleTool:
    def test_runs_optimizer(self):
        task = _task_dict()
        result = optimize_schedule.invoke({
            "target_date": "2026-11-02",
            "tasks_json": json.dumps([task]),
        })
        data = json.loads(result)
        assert "is_feasible" in data
        assert "scheduled_blocks" in data
        assert data["is_feasible"] is True

    def test_returns_blocks(self):
        task = _task_dict(estimated_mins=25)
        result = optimize_schedule.invoke({
            "target_date": "2026-11-02",
            "tasks_json": json.dumps([task]),
        })
        data = json.loads(result)
        assert len(data["scheduled_blocks"]) >= 1


class TestAllToolsList:
    def test_three_tools_registered(self):
        assert len(ALL_TOOLS) == 3
        names = {t.name for t in ALL_TOOLS}
        assert names == {"create_task", "reschedule_block", "optimize_schedule"}


# ===================================================================
# B. Prompt constants
# ===================================================================

class TestPrompts:
    def test_system_prompt_mentions_tools(self):
        assert "create_task" in SYSTEM_PROMPT
        assert "reschedule_block" in SYSTEM_PROMPT
        assert "optimize_schedule" in SYSTEM_PROMPT

    def test_router_prompt_has_both_modes(self):
        assert "control_signal" in ROUTER_PROMPT
        assert "chat" in ROUTER_PROMPT


# ===================================================================
# C. Graph state schema
# ===================================================================

class TestPlanningState:
    def test_defaults(self):
        s = PlanningState()
        assert s.entry_mode == ""
        assert s.tasks == []
        assert s.target_date == ""
        assert s.schedule_result is None
        assert s.outbound_signals is None

    def test_accepts_tasks(self):
        s = PlanningState(tasks=[_task_dict()])
        assert len(s.tasks) == 1


# ===================================================================
# D. Router node
# ===================================================================

class TestRouterNode:
    def test_pre_set_entry_mode_passthrough(self):
        s = _state(entry_mode="control_signal")
        updates = router_node(s)
        assert updates == {}  # no change

    def test_json_control_signal_detected(self):
        payload = json.dumps({"signal_type": "PREEMPTIVE_SCHEDULE_RESTRUCTURE"})
        s = _state(
            entry_mode="",
            messages=[HumanMessage(content=payload)],
        )
        updates = router_node(s)
        assert updates["entry_mode"] == "control_signal"

    def test_plain_text_classified_as_chat(self):
        s = _state(
            entry_mode="",
            messages=[HumanMessage(content="Move my study to the evening")],
        )
        updates = router_node(s)
        assert updates["entry_mode"] == "chat"

    def test_empty_messages_defaults_to_chat(self):
        s = _state(entry_mode="", messages=[])
        updates = router_node(s)
        assert updates["entry_mode"] == "chat"


# ===================================================================
# E. Control signal – PREEMPTIVE_SCHEDULE_RESTRUCTURE
# ===================================================================

class TestPreemptiveRestructure:
    def test_sets_restructure_mode(self):
        payload = json.dumps({
            "signal_type": "PREEMPTIVE_SCHEDULE_RESTRUCTURE",
            "adaptive_model_updates": {"task_delay_factor": 1.5},
        })
        s = _state(
            entry_mode="control_signal",
            messages=[HumanMessage(content=payload)],
        )
        updates = deconstructor_or_tool_llm_node(s)
        assert updates["adaptive_model"]["current_system_mode"] == "PREEMPTIVE_RESTRUCTURE"
        assert updates["adaptive_model"]["task_delay_factor"] == 1.5


# ===================================================================
# F. Control signal – Life Coach milestones
# ===================================================================

class TestLifeCoachMilestones:
    def test_ingests_milestones(self):
        payload = json.dumps({
            "signal_type": "MILESTONE_TASKS",
            "milestones": [
                {"task_id": "lc1", "title": "Study ML basics", "estimated_mins": 25,
                 "priority_weight": 3.0, "cognitive_load": "high"},
            ],
        })
        s = _state(
            entry_mode="control_signal",
            messages=[HumanMessage(content=payload)],
            tasks=[],
        )
        updates = deconstructor_or_tool_llm_node(s)
        assert len(updates["tasks"]) >= 1
        assert updates["tasks"][0]["source"] == "life_coach"

    def test_large_milestone_deconstructed(self):
        """A milestone > focus duration should be split into micro-tasks."""
        payload = json.dumps({
            "signal_type": "MILESTONE_TASKS",
            "milestones": [
                {"task_id": "lc_big", "title": "Complete stats module",
                 "estimated_mins": 80, "priority_weight": 3.0,
                 "cognitive_load": "medium"},
            ],
        })
        s = _state(
            entry_mode="control_signal",
            messages=[HumanMessage(content=payload)],
            tasks=[],
            adaptive_model={"effective_focus_duration_mins": 25},
        )
        updates = deconstructor_or_tool_llm_node(s)
        # 80 mins / 25 focus → at least 3 sub-tasks
        assert len(updates["tasks"]) >= 3
        # Total minutes should equal original
        total = sum(t["estimated_mins"] for t in updates["tasks"])
        assert total == 80


# ===================================================================
# G. Control signal – Assessment Agent remediation
# ===================================================================

class TestRemediationSignal:
    def test_ingests_remediation(self):
        payload = json.dumps({
            "signal_type": "REMEDIATION_TASKS",
            "remediation_topics": [
                {"task_id": "rem1", "title": "Probability gaps",
                 "estimated_mins": 30, "priority_weight": 4.5,
                 "cognitive_load": "high", "topic_ref": "prob_ch3"},
            ],
        })
        s = _state(
            entry_mode="control_signal",
            messages=[HumanMessage(content=payload)],
            tasks=[],
        )
        updates = deconstructor_or_tool_llm_node(s)
        assert len(updates["tasks"]) == 1
        t = updates["tasks"][0]
        assert t["source"] == "assessment_remediation"
        assert t["priority_weight"] == 4.5
        assert t["topic_ref"] == "prob_ch3"


# ===================================================================
# H. Optimizer & GCal execution node
# ===================================================================

class TestOptimizerAndGCalNode:
    def test_produces_schedule_result(self):
        s = _state()
        updates = optimizer_and_gcal_node(s)
        assert updates["schedule_result"] is not None
        sr = updates["schedule_result"]
        assert sr["is_feasible"] is True
        assert len(sr["scheduled_blocks"]) >= 1

    def test_updates_task_status_to_scheduled(self):
        s = _state()
        updates = optimizer_and_gcal_node(s)
        for t in updates["tasks"]:
            if t["task_id"] == "t1":
                assert t["status"] == "scheduled"
                assert t["scheduled_date"] == "2026-11-02"
                assert t["scheduled_start"] is not None

    def test_skips_when_no_target_date(self):
        s = _state(target_date="")
        updates = optimizer_and_gcal_node(s)
        assert updates == {}

    def test_gcal_events_created(self):
        s = _state()
        updates = optimizer_and_gcal_node(s)
        # In mock mode events should be created
        assert isinstance(updates["gcal_events_created"], list)


# ===================================================================
# I. Telemetry & outbound signals node
# ===================================================================

class TestTelemetryAndSignalsNode:
    def test_produces_outbound_signals(self):
        # First run optimizer to get schedule_result
        s = _state()
        opt_updates = optimizer_and_gcal_node(s)
        s2 = _state(
            schedule_result=opt_updates["schedule_result"],
            tasks=opt_updates["tasks"],
        )
        updates = telemetry_and_signals_node(s2)
        assert updates["outbound_signals"] is not None
        assert "workload_density_metrics" in updates["outbound_signals"]

    def test_skips_when_no_schedule_result(self):
        s = _state(schedule_result=None)
        updates = telemetry_and_signals_node(s)
        assert updates == {}

    def test_response_text_contains_density(self):
        s = _state()
        opt = optimizer_and_gcal_node(s)
        s2 = _state(
            schedule_result=opt["schedule_result"],
            tasks=opt["tasks"],
        )
        updates = telemetry_and_signals_node(s2)
        assert "Workload density" in updates["response_text"]


# ===================================================================
# J. Full graph – control signal end-to-end
# ===================================================================

class TestFullGraphControlSignal:
    def test_preemptive_restructure_e2e(self):
        graph = build_graph()
        payload = json.dumps({
            "signal_type": "PREEMPTIVE_SCHEDULE_RESTRUCTURE",
            "adaptive_model_updates": {"task_delay_factor": 1.3},
        })
        initial = {
            "messages": [HumanMessage(content=payload)],
            "entry_mode": "control_signal",
            "target_date": "2026-11-02",
            "tasks": [_task_dict(estimated_mins=50)],
            "adaptive_model": AdaptiveModelContext().model_dump(),
        }
        result = graph.invoke(initial)
        assert result["schedule_result"] is not None
        assert result["outbound_signals"] is not None

    def test_remediation_e2e(self):
        graph = build_graph()
        payload = json.dumps({
            "signal_type": "REMEDIATION_TASKS",
            "remediation_topics": [
                {"task_id": "rem1", "title": "Fix gaps",
                 "estimated_mins": 25, "priority_weight": 4.5,
                 "cognitive_load": "high"},
            ],
        })
        initial = {
            "messages": [HumanMessage(content=payload)],
            "entry_mode": "control_signal",
            "target_date": "2026-11-02",
            "tasks": [],
            "adaptive_model": AdaptiveModelContext().model_dump(),
        }
        result = graph.invoke(initial)
        # Remediation task should be scheduled
        assert result["schedule_result"]["is_feasible"] is True
        sched_ids = [b["task_id"] for b in result["schedule_result"]["scheduled_blocks"]]
        assert "rem1" in sched_ids

    def test_milestone_e2e(self):
        graph = build_graph()
        payload = json.dumps({
            "signal_type": "MILESTONE_TASKS",
            "milestones": [
                {"task_id": "lc1", "title": "Read chapter",
                 "estimated_mins": 25, "priority_weight": 2.0,
                 "cognitive_load": "low"},
            ],
        })
        initial = {
            "messages": [HumanMessage(content=payload)],
            "entry_mode": "control_signal",
            "target_date": "2026-11-02",
            "tasks": [],
            "adaptive_model": AdaptiveModelContext().model_dump(),
        }
        result = graph.invoke(initial)
        assert result["schedule_result"]["is_feasible"] is True
        assert len(result["schedule_result"]["scheduled_blocks"]) >= 1


# ===================================================================
# K. Full graph – chat mode fallback (no API key)
# ===================================================================

class TestFullGraphChatFallback:
    def test_chat_without_api_key(self, monkeypatch):
        """Without GEMINI_API_KEY the chat path should still complete."""
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "")
        graph = build_graph()
        initial = {
            "messages": [HumanMessage(content="Create a study task for math")],
            "entry_mode": "chat",
            "target_date": "2026-11-02",
            "tasks": [_task_dict()],
            "adaptive_model": AdaptiveModelContext().model_dump(),
        }
        result = graph.invoke(initial)
        # Should not crash; should have messages
        assert len(result["messages"]) >= 2  # user + at least one AI


# ===================================================================
# L. Graph build & compile
# ===================================================================

class TestGraphBuild:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_correct_nodes(self):
        """The compiled graph should contain all four nodes."""
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"router", "deconstructor_or_tool_llm",
                    "optimizer_and_gcal", "telemetry_and_signals"}
        # LangGraph adds __start__ and __end__ nodes
        assert expected.issubset(node_names)
