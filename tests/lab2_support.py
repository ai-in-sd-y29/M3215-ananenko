"""Детерминированные заглушки внешних границ для тестов Lab 2."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from mcp.shared.memory import create_connected_server_and_client_session

from app.agent.mcp_client import MCPToolbox
from app.llm.client import ChatResult, ToolCall
from app.mcp_server.errors import ToolError
from app.mcp_server.schedule import parse_schedule
from app.mcp_server.server import ServerDeps, build_server
from app.mcp_server.weather import FORECAST_URL, GEOCODING_URL
from app.services.pending import PendingAction

OWNER_SECRET = "test-owner-secret"
FIXED_NOW = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)  # 09:00 Europe/Moscow, понедельник

KAZAN_GEOCODING = {
    "results": [
        {
            "name": "Казань",
            "latitude": 55.79,
            "longitude": 49.12,
            "feature_code": "PPLA",
            "country": "Россия",
            "country_code": "RU",
            "admin1": "Татарстан",
            "population": 1243500,
        },
        {
            "name": "Казань",
            "latitude": 58.1,
            "longitude": 50.2,
            "feature_code": "PPL",
            "country": "Россия",
            "admin1": "Кировская Область",
        },
    ]
}

PARIS_GEOCODING = {
    "results": [
        {
            "name": "Париж",
            "latitude": 48.85,
            "longitude": 2.35,
            "feature_code": "PPLC",
            "country": "Франция",
            "admin1": "Иль-де-Франс",
            "population": 2138551,
        },
        {
            "name": "Париж",
            "latitude": 33.66,
            "longitude": -95.55,
            "feature_code": "PPLA2",
            "country": "США",
            "admin1": "Техас",
            "population": 24782,
        },
    ]
}

FORECAST = {
    "utc_offset_seconds": 10800,
    "timezone": "Europe/Moscow",
    "current_units": {"temperature_2m": "°C", "wind_speed_10m": "m/s"},
    "current": {
        "time": "2026-10-12T09:00",
        "temperature_2m": 7.4,
        "weather_code": 61,
        "wind_speed_10m": 3.2,
    },
}

SCHEDULE_DATA = {
    "valid_from": "2026-09-01",
    "valid_to": "2027-01-31",
    "cycle_start": "2026-08-31",
    "weeks": [
        {
            "name": "нечётная",
            "days": {
                "monday": [
                    {
                        "start": "11:40",
                        "end": "13:10",
                        "title": "Базы данных",
                        "location": "ауд. 2",
                    },
                    {
                        "start": "10:00",
                        "end": "11:30",
                        "title": "Математика",
                        "location": "ауд. 1",
                    },
                ],
                "friday": [
                    {"start": "08:20", "end": "09:50", "title": "Физика", "location": "ауд. 3"}
                ],
            },
        },
        {
            "name": "чётная",
            "days": {
                "tuesday": [
                    {
                        "start": "10:00",
                        "end": "11:30",
                        "title": "Английский",
                        "location": "ауд. 4",
                    }
                ]
            },
        },
    ],
}


class FakeFetcher:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get_json(self, url: str, params: dict[str, Any]) -> Any:
        self.calls.append((url, params))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def weather_fetcher(geocoding: Any = None, forecast: Any = None) -> FakeFetcher:
    return FakeFetcher(
        {
            GEOCODING_URL: KAZAN_GEOCODING if geocoding is None else geocoding,
            FORECAST_URL: FORECAST if forecast is None else forecast,
        }
    )


class MemoryReminderStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def insert_once(
        self, *, chat_id: int, text: str, remind_at: datetime, timezone: str, key: str
    ) -> tuple[int, bool, dict[str, Any]]:
        if key in self.rows:
            row = self.rows[key]
            if row["chat_id"] != chat_id:
                raise ToolError("owner_invalid", "Действие принадлежит другому владельцу.")
            return row["id"], False, row
        row = {
            "id": len(self.rows) + 1,
            "chat_id": chat_id,
            "text": text,
            "remind_at": remind_at,
            "timezone": timezone,
        }
        self.rows[key] = row
        return row["id"], True, row


def make_deps(
    *,
    fetcher: Any = None,
    store: Any = None,
    schedule: dict[str, Any] | None = None,
    clock: Any = None,
    secret: str = OWNER_SECRET,
) -> ServerDeps:
    parsed = parse_schedule(SCHEDULE_DATA if schedule is None else schedule)
    return ServerDeps(
        fetcher=fetcher or weather_fetcher(),
        load_schedule=lambda: parsed,
        reminder_store=store or MemoryReminderStore(),
        owner_secret=secret,
        schedule_timezone="Europe/Moscow",
        clock=clock or (lambda: FIXED_NOW),
    )


def memory_connector(deps: ServerDeps):
    server = build_server(deps)

    @asynccontextmanager
    async def connect():
        async with create_connected_server_and_client_session(server) as session:
            yield session

    return connect


@asynccontextmanager
async def running_toolbox(deps: ServerDeps):
    toolbox = MCPToolbox(memory_connector(deps), call_timeout_seconds=10)
    await toolbox.start()
    try:
        yield toolbox
    finally:
        await toolbox.stop()


def tool_call(name: str, arguments: dict[str, Any] | str, call_id: str = "call_1") -> ChatResult:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return ChatResult(content="", tool_calls=[ToolCall(id=call_id, name=name, arguments=raw)])


class ScriptedLLM:
    """Возвращает заранее заданные решения и запоминает, что видела модель."""

    def __init__(self, *results: ChatResult) -> None:
        self.results = list(results)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, messages, *, temperature, tools=None, request_id=None) -> ChatResult:
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools or []})
        return self.results.pop(0)


class FakePending:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, *, chat_id, source_key, tool_name, arguments, timezone):
        self.created.append(
            {
                "chat_id": chat_id,
                "source_key": source_key,
                "tool_name": tool_name,
                "arguments": arguments,
                "timezone": timezone,
            }
        )
        action = PendingAction(
            id="a" * 32,
            chat_id=chat_id,
            tool_name=tool_name,
            arguments=arguments,
            timezone=timezone,
            status="pending",
            result=None,
            expires_at=FIXED_NOW,
        )
        return action, True
