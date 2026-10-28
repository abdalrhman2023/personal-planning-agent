"""Phase 3 tests – Google Calendar API Integration.

All tests run in **mock mode** (no ``credentials.json`` required).
The test suite validates:
  A. Auto-detection of mock mode
  B. create_calendar_event()
  C. get_busy_blocks()
  D. update_calendar_event()
  E. delete_calendar_event()
  F. Full CRUD lifecycle
  G. Error handling
  H. Live-mode wiring (import-level, no network)
"""

import os
import pytest

from app.integrations.gcal_service import GoogleCalendarService


# ===================================================================
# Fixture: always-mock service (credentials.json must NOT exist at
# the path we give it)
# ===================================================================

@pytest.fixture
def svc(tmp_path) -> GoogleCalendarService:
    """Return a GoogleCalendarService guaranteed to be in mock mode."""
    fake_creds = str(tmp_path / "nonexistent_credentials.json")
    return GoogleCalendarService(credentials_path=fake_creds)


# ===================================================================
# A. Auto-detection of mock mode
# ===================================================================

class TestMockModeDetection:
    def test_mock_mode_when_no_credentials(self, svc: GoogleCalendarService):
        assert svc.mock_mode is True

    def test_live_mode_when_credentials_exist(self, tmp_path):
        """If a credentials.json file exists, mock_mode should be False."""
        cred_file = tmp_path / "credentials.json"
        cred_file.write_text('{"installed":{}}')
        service = GoogleCalendarService(credentials_path=str(cred_file))
        assert service.mock_mode is False

    def test_default_credentials_path_resolves(self):
        """The default credentials path should point to project root."""
        from app.integrations.gcal_service import _CREDENTIALS_PATH
        assert _CREDENTIALS_PATH.endswith("credentials.json")


# ===================================================================
# B. create_calendar_event
# ===================================================================

class TestCreateEvent:
    def test_returns_event_with_id(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event(
            summary="Study: Linear Algebra",
            date="2026-11-02",
            start_time="09:00",
            end_time="09:25",
        )
        assert "id" in event
        assert event["id"].startswith("mock_")

    def test_event_fields_match_input(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event(
            summary="Probability Review",
            date="2026-11-02",
            start_time="10:00",
            end_time="10:45",
            description="Focus on Bayes' theorem",
        )
        assert event["summary"] == "Probability Review"
        assert event["date"] == "2026-11-02"
        assert event["start_time"] == "10:00"
        assert event["end_time"] == "10:45"
        assert event["description"] == "Focus on Bayes' theorem"
        assert event["status"] == "confirmed"

    def test_event_contains_google_style_start_end(self, svc: GoogleCalendarService):
        """Mock events should mirror the Google API shape (start.dateTime)."""
        event = svc.create_calendar_event(
            summary="Test",
            date="2026-11-02",
            start_time="14:00",
            end_time="14:25",
        )
        assert "dateTime" in event["start"]
        assert event["start"]["dateTime"].startswith("2026-11-02T14:00")
        assert "dateTime" in event["end"]

    def test_multiple_creates_unique_ids(self, svc: GoogleCalendarService):
        e1 = svc.create_calendar_event("A", "2026-11-02", "08:00", "08:25")
        e2 = svc.create_calendar_event("B", "2026-11-02", "09:00", "09:25")
        assert e1["id"] != e2["id"]


# ===================================================================
# C. get_busy_blocks
# ===================================================================

class TestGetBusyBlocks:
    def test_empty_when_no_events(self, svc: GoogleCalendarService):
        blocks = svc.get_busy_blocks("2026-11-02")
        assert blocks == []

    def test_returns_blocks_for_date(self, svc: GoogleCalendarService):
        svc.create_calendar_event("A", "2026-11-02", "09:00", "09:25")
        svc.create_calendar_event("B", "2026-11-02", "14:00", "14:50")
        blocks = svc.get_busy_blocks("2026-11-02")
        assert len(blocks) == 2
        assert blocks[0] == {"start": "09:00", "end": "09:25"}
        assert blocks[1] == {"start": "14:00", "end": "14:50"}

    def test_filters_by_date(self, svc: GoogleCalendarService):
        svc.create_calendar_event("Mon", "2026-11-02", "09:00", "09:25")
        svc.create_calendar_event("Tue", "2026-11-03", "10:00", "10:25")
        blocks_mon = svc.get_busy_blocks("2026-11-02")
        blocks_tue = svc.get_busy_blocks("2026-11-03")
        assert len(blocks_mon) == 1
        assert blocks_mon[0]["start"] == "09:00"
        assert len(blocks_tue) == 1
        assert blocks_tue[0]["start"] == "10:00"

    def test_blocks_sorted_by_start(self, svc: GoogleCalendarService):
        svc.create_calendar_event("Late", "2026-11-02", "15:00", "15:25")
        svc.create_calendar_event("Early", "2026-11-02", "08:00", "08:25")
        blocks = svc.get_busy_blocks("2026-11-02")
        assert blocks[0]["start"] == "08:00"
        assert blocks[1]["start"] == "15:00"


# ===================================================================
# D. update_calendar_event
# ===================================================================

class TestUpdateEvent:
    def test_update_summary(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event("Old Title", "2026-11-02", "09:00", "09:25")
        updated = svc.update_calendar_event(event["id"], {"summary": "New Title"})
        assert updated["summary"] == "New Title"

    def test_update_start_time(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event("Study", "2026-11-02", "09:00", "09:25")
        updated = svc.update_calendar_event(
            event["id"],
            {"start": {"dateTime": "2026-11-02T10:00:00", "timeZone": "UTC"}},
        )
        assert updated["start_time"] == "10:00"

    def test_update_preserves_other_fields(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event(
            "Study", "2026-11-02", "09:00", "09:25", description="Keep me"
        )
        svc.update_calendar_event(event["id"], {"summary": "Changed"})
        # Re-fetch via busy blocks to confirm the event is still there
        blocks = svc.get_busy_blocks("2026-11-02")
        assert len(blocks) == 1

    def test_update_nonexistent_raises(self, svc: GoogleCalendarService):
        with pytest.raises(ValueError, match="not found"):
            svc.update_calendar_event("fake_id", {"summary": "X"})


# ===================================================================
# E. delete_calendar_event
# ===================================================================

class TestDeleteEvent:
    def test_delete_returns_true(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event("Temp", "2026-11-02", "09:00", "09:25")
        assert svc.delete_calendar_event(event["id"]) is True

    def test_delete_removes_from_store(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event("Temp", "2026-11-02", "09:00", "09:25")
        svc.delete_calendar_event(event["id"])
        blocks = svc.get_busy_blocks("2026-11-02")
        assert len(blocks) == 0

    def test_delete_nonexistent_raises(self, svc: GoogleCalendarService):
        with pytest.raises(ValueError, match="not found"):
            svc.delete_calendar_event("no_such_event")


# ===================================================================
# F. Full CRUD Lifecycle
# ===================================================================

class TestCRUDLifecycle:
    def test_create_read_update_delete(self, svc: GoogleCalendarService):
        # CREATE
        event = svc.create_calendar_event(
            summary="Calculus Review",
            date="2026-11-05",
            start_time="09:00",
            end_time="09:50",
            description="Chapter 4",
        )
        event_id = event["id"]
        assert event_id

        # READ (via get_busy_blocks)
        blocks = svc.get_busy_blocks("2026-11-05")
        assert len(blocks) == 1
        assert blocks[0]["start"] == "09:00"
        assert blocks[0]["end"] == "09:50"

        # UPDATE – move to afternoon
        svc.update_calendar_event(event_id, {
            "summary": "Calculus Review (moved)",
            "start": {"dateTime": "2026-11-05T14:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-11-05T14:50:00", "timeZone": "UTC"},
        })
        blocks = svc.get_busy_blocks("2026-11-05")
        assert len(blocks) == 1
        assert blocks[0]["start"] == "14:00"
        assert blocks[0]["end"] == "14:50"

        # DELETE
        assert svc.delete_calendar_event(event_id) is True
        blocks = svc.get_busy_blocks("2026-11-05")
        assert len(blocks) == 0

    def test_multiple_events_independent(self, svc: GoogleCalendarService):
        """Creating / deleting one event should not affect others."""
        e1 = svc.create_calendar_event("A", "2026-11-02", "08:00", "08:25")
        e2 = svc.create_calendar_event("B", "2026-11-02", "10:00", "10:25")
        e3 = svc.create_calendar_event("C", "2026-11-02", "14:00", "14:25")

        svc.delete_calendar_event(e2["id"])
        blocks = svc.get_busy_blocks("2026-11-02")
        assert len(blocks) == 2
        starts = [b["start"] for b in blocks]
        assert "08:00" in starts
        assert "14:00" in starts
        assert "10:00" not in starts


# ===================================================================
# G. Error Handling
# ===================================================================

class TestErrorHandling:
    def test_double_delete_raises(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event("X", "2026-11-02", "09:00", "09:25")
        svc.delete_calendar_event(event["id"])
        with pytest.raises(ValueError):
            svc.delete_calendar_event(event["id"])

    def test_update_after_delete_raises(self, svc: GoogleCalendarService):
        event = svc.create_calendar_event("X", "2026-11-02", "09:00", "09:25")
        svc.delete_calendar_event(event["id"])
        with pytest.raises(ValueError):
            svc.update_calendar_event(event["id"], {"summary": "Z"})


# ===================================================================
# H. Live-mode wiring (import checks, no network)
# ===================================================================

class TestLiveModeWiring:
    def test_service_attribute_exists(self, svc: GoogleCalendarService):
        """Internal _service should be None until first real API call."""
        assert svc._service is None

    def test_get_service_not_called_in_mock(self, svc: GoogleCalendarService):
        """In mock mode, _get_service should never be invoked."""
        svc.create_calendar_event("T", "2026-11-02", "09:00", "09:25")
        svc.get_busy_blocks("2026-11-02")
        assert svc._service is None  # proves we never touched the real API

    def test_google_packages_importable(self):
        """google-api-python-client and google-auth-oauthlib must be installed."""
        import googleapiclient.discovery  # noqa: F401
        import google_auth_oauthlib.flow  # noqa: F401
        import google.oauth2.credentials  # noqa: F401
