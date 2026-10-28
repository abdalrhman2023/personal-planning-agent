"""Phase 5 tests – FastAPI API Server and Endpoints.

Tests cover:
  A. Health check endpoint function
  B. Schedule optimization endpoint (POST /api/v1/planning/optimize)
  C. Telemetry endpoint (POST /api/v1/planning/telemetry)
  D. Chat endpoint (POST /api/v1/planning/chat) with control signal
  E. FastAPI app configuration & route registration
"""

import pytest
from app.api.routes import (
    ChatRequest,
    ChatResponse,
    OptimizeResponse,
    TelemetryRequest,
    chat_endpoint,
    optimize_schedule_endpoint,
    router,
    telemetry_endpoint,
)
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    OutboundSignals,
    ScheduleRequest,
    ScheduledBlock,
    StudyTask,
)
from main import app, health_check


class TestHealthCheck:
    def test_health_check_returns_online(self):
        data = health_check()
        assert data["status"] == "online"
        assert "Personal Planning Agent" in data["agent"]


class TestAppRegistration:
    def test_routes_registered_on_app(self):
        router_paths = [r.path for r in router.routes]
        assert "/api/v1/planning/optimize" in router_paths
        assert "/api/v1/planning/chat" in router_paths
        assert "/api/v1/planning/telemetry" in router_paths


class TestOptimizeEndpoint:
    def test_optimize_schedule_success(self):
        req = ScheduleRequest(
            target_date="2026-11-02",
            tasks=[
                StudyTask(
                    task_id="api_t1",
                    title="API Testing Study",
                    source="user",
                    estimated_mins=25,
                    priority_weight=3.0,
                    cognitive_load="medium",
                    status="pending",
                )
            ],
            university_schedule=[
                FixedLecture(
                    id="lec_1",
                    day="Monday",
                    date="2026-11-02",
                    start="09:00",
                    end="10:30",
                    title="Software Architecture",
                )
            ],
            gcal_busy_blocks=[],
            adaptive_model=AdaptiveModelContext(
                peak_productivity_window="10:30-12:00",
                secondary_productivity_window="18:00-21:00",
                effective_focus_duration_mins=25,
                max_daily_study_mins=240,
                current_system_mode="NORMAL",
            ),
        )

        resp = optimize_schedule_endpoint(req)
        assert isinstance(resp, OptimizeResponse)
        assert resp.is_feasible is True
        assert len(resp.scheduled_blocks) == 1
        assert len(resp.gcal_events) == 1
        assert resp.outbound_signals is not None
        assert "workload_density" in resp.outbound_signals.workload_density_metrics

    def test_optimize_schedule_with_nested_study_data(self):
        """Verify ScheduleRequest accepts nested study_data {tasks, university_schedule} per spec."""
        payload = {
            "target_date": "2026-11-02",
            "study_data": {
                "tasks": [
                    {
                        "task_id": "nested_t1",
                        "title": "Nested Study Task",
                        "source": "life_coach",
                        "estimated_mins": 25,
                        "priority_weight": 3.0,
                        "cognitive_load": "medium",
                        "status": "pending",
                    }
                ],
                "university_schedule": [
                    {
                        "id": "lec_nested",
                        "day": "Monday",
                        "date": "2026-11-02",
                        "start": "09:00",
                        "end": "10:00",
                        "title": "Machine Learning",
                    }
                ],
            },
            "gcal_busy_blocks": [],
        }
        req = ScheduleRequest.model_validate(payload)
        assert len(req.tasks) == 1
        assert req.tasks[0].task_id == "nested_t1"
        assert len(req.university_schedule) == 1
        assert req.university_schedule[0].title == "Machine Learning"

        resp = optimize_schedule_endpoint(req)
        assert isinstance(resp, OptimizeResponse)
        assert resp.is_feasible is True
        assert len(resp.scheduled_blocks) == 1


class TestTelemetryEndpoint:
    def test_telemetry_computation(self):
        req = TelemetryRequest(
            target_date="2026-11-02",
            scheduled_blocks=[
                ScheduledBlock(
                    task_id="t1",
                    title="Study Block 1",
                    date="2026-11-02",
                    start="11:00",
                    end="11:25",
                    part=1,
                )
            ],
            tasks=[
                StudyTask(
                    task_id="t1",
                    title="Study Block 1",
                    source="user",
                    estimated_mins=25,
                    priority_weight=3.0,
                    cognitive_load="medium",
                    status="completed",
                )
            ],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        resp = telemetry_endpoint(req)
        assert isinstance(resp, OutboundSignals)
        assert "workload_density" in resp.workload_density_metrics
        assert len(resp.to_ai_tutor) == 1


class TestChatEndpoint:
    def test_control_signal_chat(self):
        req = ChatRequest(
            message='{"signal_type": "PREEMPTIVE_SCHEDULE_RESTRUCTURE", "adaptive_model_updates": {"task_delay_factor": 1.2}}',
            entry_mode="control_signal",
            target_date="2026-11-02",
            tasks=[],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        resp = chat_endpoint(req)
        assert isinstance(resp, ChatResponse)
        assert resp.entry_mode == "control_signal"
        assert "PREEMPTIVE_SCHEDULE_RESTRUCTURE" in resp.response

    def test_chat_endpoint_response_time_under_5_seconds(self):
        import time

        req = ChatRequest(
            message="I want to study linear algebra chapter 3 for 50 minutes tomorrow.",
            entry_mode="chat",
            target_date="2026-11-02",
            tasks=[],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        start = time.time()
        resp = chat_endpoint(req)
        elapsed = time.time() - start

        assert isinstance(resp, ChatResponse)
        assert resp.response is not None
        assert elapsed < 5.0, f"Chat endpoint took {elapsed:.2f}s, expected < 5.0s"

    def test_chat_natural_language_task_creation_and_optimization(self):
        """Verify chat mode parses natural language request, creates/deconstructs tasks,
        optimizes the schedule, and syncs calendar events without returning empty or null."""
        req = ChatRequest(
            message="Add a high priority study task for Deep Learning review tomorrow for 45 minutes and optimize my schedule",
            entry_mode="chat",
            target_date="2026-11-02",
            tasks=[],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        resp = chat_endpoint(req)
        assert isinstance(resp, ChatResponse)
        assert len(resp.tasks) >= 1
        assert resp.schedule_result is not None
        assert len(resp.schedule_result.scheduled_blocks) >= 1
        assert len(resp.gcal_events) >= 1
        assert resp.outbound_signals is not None
        # Tasks should have been deconstructed because 45 mins > default 25 min focus limit
        assert any("Deep Learning" in t.title for t in resp.tasks)
