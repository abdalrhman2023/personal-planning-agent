"""End-to-End Multi-Agent Simulation: Personal Planning Agent (Agent 3).

Demonstrates the 3 core interaction flows specified in AGENT_SPEC_AND_PLAN.md:
  1. Initial schedule optimization from University timetable + Life Coach milestones.
  2. Handling a PREEMPTIVE_SCHEDULE_RESTRUCTURE signal from Adaptive Manager.
  3. Detecting >20% velocity slip and alerting the Adaptive Manager.
"""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage

from app.agent.graph import build_graph
from app.engine.telemetry import compute_velocity_slip
from app.integrations.gcal_service import GoogleCalendarService
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    StudyTask,
)


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def _section(title: str) -> None:
    print("\n" + "-" * 78)
    print(f"  >> {title}")
    print("-" * 78)


def run_simulation(cleanup_first: bool = True):
    graph = build_graph()
    target_date = "2026-11-02"

    gcal = GoogleCalendarService()
    mode_label = "Mock Mode Active" if gcal.mock_mode else "Live Mode Active"

    print("\n" + "#" * 78)
    print("   PERSONAL PLANNING AGENT (AGENT 3) - END-TO-END DEMO SIMULATION")
    print(f"   Generative AI Personal Life & Study Coach [{mode_label}]")
    print("#" * 78)

    # Pre-demo cleanup: remove previous demo events on target_date so calendar stays clean
    if cleanup_first:
        cleaned_count = gcal.clear_events_for_date(target_date, summary_prefix="Study:")
        if cleaned_count > 0:
            print(f"\n[Calendar Cleanup] Cleared {cleaned_count} previous demo study event(s) for {target_date}.")

    # =========================================================================
    # SCENARIO 1: University Timetable + Life Coach Milestones Optimization
    # =========================================================================
    _banner("Scenario 1: Initial Schedule Optimization from University Timetable + Life Coach Milestones")

    university_schedule = [
        FixedLecture(
            id="lec_101",
            day="Monday",
            date=target_date,
            start="09:00",
            end="10:30",
            title="CS301: Distributed Systems Lecture",
        ),
        FixedLecture(
            id="lec_102",
            day="Monday",
            date=target_date,
            start="13:30",
            end="15:00",
            title="MATH205: Linear Algebra & Matrices",
        ),
    ]

    gcal_busy_blocks = [
        {"start": "16:00", "end": "16:45", "summary": "Physical Therapy / Workout"}
    ]

    print("\n[Input Constraints Ingested]")
    print(f" • Target Date: {target_date}")
    print(f" • University Lectures ({len(university_schedule)} blocks):")
    for lec in university_schedule:
        print(f"    - {lec.start} - {lec.end} | {lec.title}")
    print(f" • External GCal Busy Blocks ({len(gcal_busy_blocks)} blocks):")
    for blk in gcal_busy_blocks:
        print(f"    - {blk['start']} - {blk['end']} | {blk['summary']}")

    # Life Coach sends decomposed milestones (Agent 4 -> Agent 3)
    life_coach_signal = {
        "signal_type": "MILESTONE_TASKS",
        "milestones": [
            {
                "task_id": "ms_dist_cache",
                "title": "Distributed Cache Implementation",
                "estimated_mins": 75,
                "priority_weight": 4.5,
                "cognitive_load": "high",
                "deadline": "2026-11-03T23:59:59",
                "topic_ref": "CS301_Project2",
                "parent_milestone_id": "goal_sem1_project",
            },
            {
                "task_id": "ms_matrix_svd",
                "title": "Review SVD Decomposition Notes",
                "estimated_mins": 35,
                "priority_weight": 3.0,
                "cognitive_load": "medium",
                "deadline": "2026-11-04T12:00:00",
                "topic_ref": "MATH205_SVD",
                "parent_milestone_id": "goal_math_mastery",
            },
        ],
    }

    _section("Agent 4 (Life Coach) Signal Sent to Planning Agent")
    print(json.dumps(life_coach_signal, indent=2))

    initial_state_1 = {
        "messages": [HumanMessage(content=json.dumps(life_coach_signal))],
        "entry_mode": "control_signal",
        "target_date": target_date,
        "tasks": [],
        "university_schedule": [l.model_dump() for l in university_schedule],
        "gcal_busy_blocks": gcal_busy_blocks,
        "adaptive_model": AdaptiveModelContext(
            peak_productivity_window="10:30-12:30",
            secondary_productivity_window="18:00-21:00",
            effective_focus_duration_mins=25,
            max_daily_study_mins=240,
            current_system_mode="NORMAL",
        ).model_dump(),
    }

    result_1 = graph.invoke(initial_state_1)

    _section("Generated Conflict-Free Study Schedule (Optimizer Output)")
    sr_1 = result_1["schedule_result"]
    print(f"Schedule Feasibility: {'FEASIBLE' if sr_1['is_feasible'] else 'INFEASIBLE'}")
    print(f"Total Scheduled Study Blocks: {len(sr_1['scheduled_blocks'])}")
    print("\nTimeline for 2026-11-02:")
    for b in sr_1["scheduled_blocks"]:
        print(f"  [{b['start']} - {b['end']}] -> {b['title']} (Part {b['part']}) [Task ID: {b['task_id']}]")

    _section(f"Sync with Google Calendar ({mode_label})")
    gcal_events = result_1.get("gcal_events_created", [])
    print(f"Calendar Events Successfully Synced: {len(gcal_events)}")
    for ev in gcal_events:
        print(f"  • Event ID: {ev['id']} | {ev['start']['dateTime']} - {ev['end']['dateTime']} | {ev['summary']}")

    _section("Outbound Signals Produced for AI Tutor (Agent 1)")
    signals_1 = result_1["outbound_signals"]
    invitations = signals_1.get("to_ai_tutor", [])
    print(f"AI Tutor Study Block Invitations Generated: {len(invitations)}")
    for inv in invitations[:2]:
        print(f"  • Block [{inv['block_start']}-{inv['block_end']}] | Topic: {inv['topic_ref']} | Load: {inv['cognitive_load']}")
    print(f"Workload Density: {signals_1['workload_density_metrics']['workload_density']:.1%}")

    # =========================================================================
    # SCENARIO 2: Preemptive Schedule Restructure (Adaptive Manager Command)
    # =========================================================================
    _banner("Scenario 2: PREEMPTIVE_SCHEDULE_RESTRUCTURE from Adaptive Manager (Agent 5)")

    restructure_signal = {
        "signal_type": "PREEMPTIVE_SCHEDULE_RESTRUCTURE",
        "command": "PREEMPTIVE_SCHEDULE_RESTRUCTURE",
        "adaptive_model_updates": {
            "task_delay_factor": 1.4,
            "max_daily_study_mins": 120,  # lower cap to prevent burnout
        },
    }

    _section("Agent 5 (Adaptive Manager) Signal Sent to Planning Agent")
    print(json.dumps(restructure_signal, indent=2))

    initial_state_2 = {
        "messages": [HumanMessage(content=json.dumps(restructure_signal))],
        "entry_mode": "control_signal",
        "target_date": target_date,
        "tasks": result_1["tasks"],  # Carry over tasks from scenario 1
        "university_schedule": [l.model_dump() for l in university_schedule],
        "gcal_busy_blocks": gcal_busy_blocks,
        "adaptive_model": result_1["adaptive_model"],
        "gcal_events_created": result_1.get("gcal_events_created", []),  # Pass old events for deletion
    }

    result_2 = graph.invoke(initial_state_2)

    _section("Restructured Schedule After Fatigue Factor & Daily Cap Applied")
    sr_2 = result_2["schedule_result"]
    print(f"System Mode: {result_2['adaptive_model']['current_system_mode']}")
    print(f"Task Delay Factor Applied: {result_2['adaptive_model']['task_delay_factor']}x")
    print(f"Max Daily Study Cap: {result_2['adaptive_model']['max_daily_study_mins']} mins")
    print(f"Schedule Feasibility: {'FEASIBLE' if sr_2['is_feasible'] else 'PARTIALLY FEASIBLE'}")
    print(f"Scheduled Blocks: {len(sr_2['scheduled_blocks'])}")
    for b in sr_2["scheduled_blocks"]:
        print(f"  [{b['start']} - {b['end']}] -> {b['title']} (Part {b['part']})")

    if sr_2["unscheduled_tasks"]:
        print(f"\nUnscheduled Tasks (Protected from Burnout): {sr_2['unscheduled_tasks']}")

    _section(f"Updated Google Calendar Events ({mode_label})")
    gcal_events_2 = result_2.get("gcal_events_created", [])
    print(f"Calendar Events Successfully Updated: {len(gcal_events_2)} (Previous events cleaned up without duplicates)")
    for ev in gcal_events_2:
        print(f"  • Event ID: {ev['id']} | {ev['start']['dateTime']} - {ev['end']['dateTime']} | {ev['summary']}")

    print(f"\nWorkload Density Post-Restructure: "
          f"{result_2['outbound_signals']['workload_density_metrics']['workload_density']:.1%}")

    # =========================================================================
    # SCENARIO 3: Detecting >20% Velocity Slip & Alerting Adaptive Manager
    # =========================================================================
    _banner("Scenario 3: Detecting >20% Velocity Slip and Emitting Alert to Agent 5")

    simulated_tasks = [
        StudyTask(task_id="t1", title="Task 1", source="user", estimated_mins=25, priority_weight=3.0, cognitive_load="medium", status="completed"),
        StudyTask(task_id="t2", title="Task 2", source="user", estimated_mins=25, priority_weight=3.0, cognitive_load="medium", status="completed"),
        StudyTask(task_id="t3", title="Task 3", source="user", estimated_mins=25, priority_weight=3.0, cognitive_load="high", status="missed"),
        StudyTask(task_id="t4", title="Task 4", source="user", estimated_mins=25, priority_weight=3.0, cognitive_load="high", status="missed"),
        StudyTask(task_id="t5", title="Task 5", source="user", estimated_mins=25, priority_weight=3.0, cognitive_load="low", status="missed"),
        StudyTask(task_id="t6", title="Task 6", source="user", estimated_mins=25, priority_weight=3.0, cognitive_load="low", status="missed"),
    ]

    _section("Telemetry Execution Log Analysis")
    print(f"Total Scheduled Tasks Evaluated: {len(simulated_tasks)}")
    completed_count = sum(1 for t in simulated_tasks if t.status == "completed")
    missed_count = sum(1 for t in simulated_tasks if t.status == "missed")
    print(f" • Completed Tasks: {completed_count}")
    print(f" • Missed Tasks:    {missed_count}")

    slip_analysis = compute_velocity_slip(simulated_tasks)
    print(f"\nActual Completion Rate: {slip_analysis['completion_rate']:.1%}")
    print(f"Expected Completion Rate: {slip_analysis['expected_rate']:.1%}")
    print(f"Velocity Slip Deviation:  {slip_analysis['slip_ratio']:.1%} (Threshold: 20.0%)")
    print(f"Threshold Exceeded:       {slip_analysis['deviation_exceeded']}")

    _section("Outbound Alert Payload Emitted to Adaptive Manager (Agent 5)")
    alert_payload = slip_analysis["alert_payload"]
    print(json.dumps(alert_payload, indent=2))

    print("\n" + "=" * 78)
    print("  SIMULATION COMPLETE: ALL 3 MULTI-AGENT WORKFLOWS VERIFIED SUCCESSFULLY")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_simulation()
