"""Google Calendar integration service.

Provides ``GoogleCalendarService`` with two operating modes:

* **Live mode** – uses ``google-api-python-client`` and ``google-auth-oauthlib``
  when ``credentials.json`` and ``token.json`` are found in the project root.
* **Mock mode** – activates automatically when credentials/token are absent,
  keeping an in-memory event store so the entire application (and its tests)
  can run without network access or real OAuth2 tokens.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CREDENTIALS_PATH = os.path.join(_ROOT_DIR, "credentials.json")
_TOKEN_PATH = os.path.join(_ROOT_DIR, "token.json")
_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_CALENDAR_ID = "primary"


# ---------------------------------------------------------------------------
# Google Calendar Service
# ---------------------------------------------------------------------------


class GoogleCalendarService:
    """Unified interface for Google Calendar operations.

    Automatically enters **mock mode** when ``credentials.json`` or ``token.json``
    is not found. In mock mode every method behaves identically (same signatures,
    same return shapes) but operates against an in-memory dict instead of the
    real Google Calendar API.
    """

    def __init__(self, credentials_path: Optional[str] = None) -> None:
        self._credentials_path = credentials_path or _CREDENTIALS_PATH
        has_creds = os.path.isfile(self._credentials_path)
        has_token = os.path.isfile(_TOKEN_PATH)

        if credentials_path is not None:
            # Explicit credentials path passed (e.g. from tests)
            self._mock_mode = not has_creds
        elif os.getenv("PYTEST_CURRENT_TEST") or os.getenv("GCAL_FORCE_MOCK"):
            # Automated tests run in mock mode to stay offline and protect user's calendar
            self._mock_mode = True
        else:
            # Default usage: live mode requires credentials.json AND token.json
            self._mock_mode = not (has_creds and has_token)

        self._service: Any = None  # google API resource (lazy-built)

        # In-memory event store for mock mode  {event_id: event_dict}
        self._mock_events: Dict[str, dict] = {}

    # -- properties ----------------------------------------------------------

    @property
    def mock_mode(self) -> bool:
        """``True`` when running without real Google Calendar credentials."""
        return self._mock_mode

    # -- internal: lazy Google API client ------------------------------------

    def _get_service(self) -> Any:
        """Return an authenticated ``googleapiclient.discovery.Resource``.

        Performs token loading/refresh with socket timeouts; never blocks on
        interactive local servers during background requests.
        """
        if self._service is not None:
            return self._service

        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        creds: Optional[Credentials] = None

        if os.path.isfile(_TOKEN_PATH):
            try:
                creds = Credentials.from_authorized_user_file(_TOKEN_PATH, _SCOPES)
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    with open(_TOKEN_PATH, "w") as token_file:
                        token_file.write(creds.to_json())
                except Exception:
                    self._mock_mode = True
                    return None
            else:
                # Never block during server requests or tests if not pre-authorized
                self._mock_mode = True
                return None

        if creds and creds.valid:
            import socket
            socket.setdefaulttimeout(10)
            self._service = build("calendar", "v3", credentials=creds)
            return self._service

        self._mock_mode = True
        return None

    # -----------------------------------------------------------------------
    # 1. get_busy_blocks
    # -----------------------------------------------------------------------

    def get_busy_blocks(
        self,
        date: str,
        calendar_id: str = _CALENDAR_ID,
    ) -> List[Dict[str, str]]:
        """Return busy time blocks for a given *date* (``YYYY-MM-DD``).

        Each element is ``{"start": "HH:MM", "end": "HH:MM"}``.
        """
        if self._mock_mode:
            return self._mock_get_busy_blocks(date)

        service = self._get_service()
        if self._mock_mode or service is None:
            return self._mock_get_busy_blocks(date)

        tz_name = getattr(config, "TIMEZONE", "Africa/Cairo")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = None

        time_min = f"{date}T00:00:00Z"
        time_max = f"{date}T23:59:59Z"

        body = {
            "timeMin": time_min,
            "timeMax": time_max,
            "items": [{"id": calendar_id}],
        }

        try:
            result = service.freebusy().query(body=body).execute()
            busy_list = result.get("calendars", {}).get(calendar_id, {}).get("busy", [])

            blocks: List[Dict[str, str]] = []
            for entry in busy_list:
                start_dt = datetime.fromisoformat(entry["start"].replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(entry["end"].replace("Z", "+00:00"))
                if tz:
                    start_dt = start_dt.astimezone(tz)
                    end_dt = end_dt.astimezone(tz)
                blocks.append({
                    "start": start_dt.strftime("%H:%M"),
                    "end": end_dt.strftime("%H:%M"),
                })

            return blocks
        except Exception:
            return self._mock_get_busy_blocks(date)

    # -----------------------------------------------------------------------
    # 2. create_calendar_event
    # -----------------------------------------------------------------------

    def create_calendar_event(
        self,
        summary: str,
        date: str,
        start_time: str,
        end_time: str,
        description: str = "",
        calendar_id: str = _CALENDAR_ID,
    ) -> Dict[str, Any]:
        """Create a calendar event and return it (including ``id``)."""
        if self._mock_mode:
            return self._mock_create_event(summary, date, start_time, end_time, description)

        service = self._get_service()
        if self._mock_mode or service is None:
            return self._mock_create_event(summary, date, start_time, end_time, description)

        tz_name = getattr(config, "TIMEZONE", "Africa/Cairo")
        event_body = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": f"{date}T{start_time}:00",
                "timeZone": tz_name,
            },
            "end": {
                "dateTime": f"{date}T{end_time}:00",
                "timeZone": tz_name,
            },
        }

        try:
            created = (
                service.events()
                .insert(calendarId=calendar_id, body=event_body)
                .execute()
            )
            return created
        except Exception:
            return self._mock_create_event(summary, date, start_time, end_time, description)

    # -----------------------------------------------------------------------
    # 3. update_calendar_event
    # -----------------------------------------------------------------------

    def update_calendar_event(
        self,
        event_id: str,
        updates: Dict[str, Any],
        calendar_id: str = _CALENDAR_ID,
    ) -> Dict[str, Any]:
        """Patch an existing calendar event and return the updated version."""
        if self._mock_mode:
            return self._mock_update_event(event_id, updates)

        service = self._get_service()
        if self._mock_mode or service is None:
            return self._mock_update_event(event_id, updates)
        try:
            updated = (
                service.events()
                .patch(calendarId=calendar_id, eventId=event_id, body=updates)
                .execute()
            )
            return updated
        except Exception:
            return self._mock_update_event(event_id, updates)

    # -----------------------------------------------------------------------
    # 4. delete_calendar_event
    # -----------------------------------------------------------------------

    def delete_calendar_event(
        self,
        event_id: str,
        calendar_id: str = _CALENDAR_ID,
    ) -> bool:
        """Delete a calendar event. Returns ``True`` on success."""
        if self._mock_mode:
            return self._mock_delete_event(event_id)

        service = self._get_service()
        if self._mock_mode or service is None:
            return self._mock_delete_event(event_id)
        service.events().delete(
            calendarId=calendar_id, eventId=event_id
        ).execute()
        return True

    # -----------------------------------------------------------------------
    # 5. clear_events_for_date
    # -----------------------------------------------------------------------

    def clear_events_for_date(
        self,
        date: str,
        summary_prefix: str = "",
        calendar_id: str = _CALENDAR_ID,
    ) -> int:
        """Delete events on *date* matching summary_prefix. Returns count deleted."""
        if self._mock_mode:
            to_delete = [
                eid for eid, ev in self._mock_events.items()
                if ev.get("date") == date and (not summary_prefix or ev.get("summary", "").startswith(summary_prefix))
            ]
            for eid in to_delete:
                del self._mock_events[eid]
            return len(to_delete)

        service = self._get_service()
        if self._mock_mode or service is None:
            return 0

        time_min = f"{date}T00:00:00Z"
        time_max = f"{date}T23:59:59Z"
        events_result = (
            service.events()
            .list(calendarId=calendar_id, timeMin=time_min, timeMax=time_max, singleEvents=True)
            .execute()
        )
        items = events_result.get("items", [])
        deleted = 0
        for item in items:
            summary = item.get("summary", "")
            if not summary_prefix or summary.startswith(summary_prefix):
                try:
                    service.events().delete(calendarId=calendar_id, eventId=item["id"]).execute()
                    deleted += 1
                except Exception:
                    pass
        return deleted

    # ===================================================================
    # Mock-mode implementations
    # ===================================================================

    def _mock_get_busy_blocks(self, date: str) -> List[Dict[str, str]]:
        blocks: List[Dict[str, str]] = []
        for evt in self._mock_events.values():
            if evt.get("date") == date:
                blocks.append({"start": evt["start_time"], "end": evt["end_time"]})
        return sorted(blocks, key=lambda b: b["start"])

    def _mock_create_event(
        self,
        summary: str,
        date: str,
        start_time: str,
        end_time: str,
        description: str,
    ) -> Dict[str, Any]:
        tz_name = getattr(config, "TIMEZONE", "Africa/Cairo")
        event_id = f"mock_{uuid.uuid4().hex[:12]}"
        event = {
            "id": event_id,
            "summary": summary,
            "description": description,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "start": {"dateTime": f"{date}T{start_time}:00", "timeZone": tz_name},
            "end": {"dateTime": f"{date}T{end_time}:00", "timeZone": tz_name},
            "status": "confirmed",
        }
        self._mock_events[event_id] = event
        return event

    def _mock_update_event(
        self, event_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        if event_id not in self._mock_events:
            raise ValueError(f"Event '{event_id}' not found in mock store.")
        event = self._mock_events[event_id]
        event.update(updates)
        if "summary" in updates:
            event["summary"] = updates["summary"]
        if "start" in updates and "dateTime" in updates["start"]:
            dt = updates["start"]["dateTime"]
            event["start_time"] = dt[11:16]
            event["date"] = dt[:10]
        if "end" in updates and "dateTime" in updates["end"]:
            dt = updates["end"]["dateTime"]
            event["end_time"] = dt[11:16]
        return event

    def _mock_delete_event(self, event_id: str) -> bool:
        if event_id not in self._mock_events:
            raise ValueError(f"Event '{event_id}' not found in mock store.")
        del self._mock_events[event_id]
        return True
