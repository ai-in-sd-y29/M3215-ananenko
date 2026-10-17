"""Lab 2: MCP-сервер — инструменты, схемы, ресурс, владелец и идемпотентность."""

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from aiohttp import web
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from app.agent.errors import MCPToolError
from app.mcp_server.errors import ToolError
from app.mcp_server.reminders import add_reminder, owner_from_meta
from app.mcp_server.schedule import build_week, load_schedule, parse_schedule
from app.mcp_server.schemas import ReminderInput
from app.mcp_server.server import WEEK_RESOURCE_URI, build_server, current_week
from app.mcp_server.weather import (
    HttpJsonFetcher,
    get_weather,
    weather_bad_response,
    weather_timeout,
)
from app.owner_token import META_KEY, OwnerContext, sign_owner_token
from tests.lab2_support import (
    FIXED_NOW,
    OWNER_SECRET,
    PARIS_GEOCODING,
    SCHEDULE_DATA,
    MemoryReminderStore,
    make_deps,
    running_toolbox,
    weather_fetcher,
)

FUTURE = "2026-10-13T18:30:00+03:00"


def owner_meta(chat_id: int = 1, action_id: str = "action-1") -> dict[str, str]:
    token = sign_owner_token(
        OWNER_SECRET,
        OwnerContext(chat_id=chat_id, action_id=action_id, timezone="Europe/Moscow"),
    )
    return {META_KEY: token}


# --- Погода -------------------------------------------------------------------------


async def test_weather_success_returns_normalized_fields():
    # Arrange
    fetcher = weather_fetcher()
    # Act
    result = await get_weather("  Казань ", fetcher)
    # Assert
    assert result.status == "ok"
    assert result.location.name == "Казань"
    assert result.location.country == "Россия"
    assert result.location.admin1 == "Татарстан"
    assert result.current.temperature_c == 7.4
    assert result.current.description == "небольшой дождь"
    assert result.current.wind_speed == 3.2
    assert result.current.wind_speed_unit == "m/s"
    assert result.current.observed_at == "2026-10-12T09:00:00+03:00"
    assert result.current.timezone == "Europe/Moscow"
    assert fetcher.calls[1][1]["latitude"] == 55.79


async def test_weather_city_not_found():
    # Arrange
    fetcher = weather_fetcher(geocoding={"generationtime_ms": 0.5})
    # Act
    result = await get_weather("Ыыыкщ", fetcher)
    # Assert
    assert result.status == "not_found"
    assert result.current is None
    assert len(fetcher.calls) == 1


async def test_weather_ambiguous_city_returns_options_not_first():
    # Arrange
    fetcher = weather_fetcher(geocoding=PARIS_GEOCODING)
    # Act
    result = await get_weather("Париж", fetcher)
    # Assert
    assert result.status == "ambiguous"
    assert result.location is None
    assert [(o.country, o.admin1) for o in result.options] == [
        ("Франция", "Иль-де-Франс"),
        ("США", "Техас"),
    ]
    assert len(fetcher.calls) == 1


async def test_weather_country_qualifier_resolves_ambiguity():
    result = await get_weather("Париж, США", weather_fetcher(geocoding=PARIS_GEOCODING))
    assert result.status == "ok"
    assert result.location.admin1 == "Техас"


@pytest.mark.parametrize(
    ("forecast", "geocoding"),
    [
        ({"current": {"time": "bad"}}, None),
        ({"utc_offset_seconds": 0, "timezone": "UTC", "current": {}}, None),
        (None, {"results": [{"name": 5}]}),
        (None, ["not", "an", "object"]),
    ],
)
async def test_weather_broken_api_response_is_error(forecast, geocoding):
    with pytest.raises(ToolError) as error:
        await get_weather("Казань", weather_fetcher(geocoding=geocoding, forecast=forecast))
    assert error.value.code == "weather_bad_response"


async def test_weather_timeout_is_reported_via_mcp_as_structured_error():
    # Arrange
    fetcher = weather_fetcher(geocoding=weather_timeout())
    async with running_toolbox(make_deps(fetcher=fetcher)) as toolbox:
        # Act
        with pytest.raises(MCPToolError) as error:
            await toolbox.call_tool("get_weather", {"city": "Казань"})
    # Assert
    assert error.value.code == "weather_timeout"
    assert "Traceback" not in error.value.message


async def test_http_fetcher_retries_read_once_then_times_out(unused_tcp_port):
    # Arrange: первый ответ 500, второй — слишком медленный.
    hits = []

    async def handler(request):
        hits.append(1)
        if len(hits) == 1:
            return web.Response(status=500)
        await asyncio.sleep(2)
        return web.json_response({})

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    fetcher = HttpJsonFetcher(timeout_seconds=0.3, retries=1)
    try:
        # Act
        with pytest.raises(ToolError) as error:
            await fetcher.get_json(f"http://127.0.0.1:{unused_tcp_port}/", {})
    finally:
        await fetcher.aclose()
        await runner.cleanup()
    # Assert
    assert error.value.code == "weather_timeout"
    assert len(hits) == 2


async def test_http_fetcher_reports_invalid_json(unused_tcp_port):
    async def handler(request):
        return web.Response(text="<html>oops</html>", content_type="text/html")

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    fetcher = HttpJsonFetcher(timeout_seconds=2)
    try:
        with pytest.raises(ToolError) as error:
            await fetcher.get_json(f"http://127.0.0.1:{unused_tcp_port}/", {})
    finally:
        await fetcher.aclose()
        await runner.cleanup()
    assert error.value.code == weather_bad_response().code


# --- Расписание ---------------------------------------------------------------------


async def test_schedule_day_with_lessons_sorted_by_time():
    async with running_toolbox(make_deps()) as toolbox:
        result = await toolbox.call_tool("get_schedule", {"date": "2026-10-12"})
    assert result["weekday"] == "понедельник"
    assert result["timezone"] == "Europe/Moscow"
    assert result["lessons"] == [
        {"start": "10:00", "end": "11:30", "title": "Математика", "location": "ауд. 1"},
        {"start": "11:40", "end": "13:10", "title": "Базы данных", "location": "ауд. 2"},
    ]


async def test_schedule_day_without_lessons_is_empty_list():
    async with running_toolbox(make_deps()) as toolbox:
        sunday = await toolbox.call_tool("get_schedule", {"date": "2026-10-18"})
        even_tuesday = await toolbox.call_tool("get_schedule", {"date": "2026-10-13"})
        odd_tuesday = await toolbox.call_tool("get_schedule", {"date": "2026-10-20"})
    assert sunday["lessons"] == []
    assert even_tuesday["lessons"] == []
    assert odd_tuesday["lessons"][0]["title"] == "Английский"


@pytest.mark.parametrize("raw", ["12.10.2026", "2026-13-01", "2026-02-30", "завтра"])
async def test_schedule_invalid_date_is_rejected(raw):
    async with running_toolbox(make_deps()) as toolbox:
        with pytest.raises(MCPToolError) as error:
            await toolbox.call_tool("get_schedule", {"date": raw})
    assert error.value.code == "invalid_arguments"


def test_repository_schedule_covers_two_weeks_without_personal_data():
    schedule = load_schedule("data/schedule.json")
    assert len(schedule.weeks) >= 2
    # 19.10.2026 — понедельник нечётной недели; 26.10 — понедельник чётной.
    odd_monday = schedule.lessons_on(date(2026, 10, 19))
    even_monday = schedule.lessons_on(date(2026, 10, 26))
    odd_thursday = schedule.lessons_on(date(2026, 10, 22))
    even_thursday = schedule.lessons_on(date(2026, 10, 29))
    assert odd_monday and even_monday and odd_monday == even_monday
    assert odd_thursday == []  # фитнес в четверг на нечётной убран
    assert any("Фитнес" in lesson.title for lesson in even_thursday)
    assert any("История русской культуры" in lesson.title for lesson in even_thursday)
    raw = Path("data/schedule.json").read_text(encoding="utf-8")
    assert "Тертычный" not in raw and "Родикова" not in raw


def test_relative_calendar_nearest_sunday_from_saturday():
    from app.agent.loop import format_relative_calendar

    calendar = format_relative_calendar(date(2026, 10, 17))
    assert "сегодня: 2026-10-17 (суббота)" in calendar
    assert "ближайший воскресенье: 2026-10-18" in calendar
    assert "2026-10-23" not in calendar or "пятница" in calendar
    assert "ближайший воскресенье: 2026-10-23" not in calendar


# --- Ресурс недели ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("today", "start", "end"),
    [
        (date(2026, 10, 1), "2026-09-28", "2026-10-04"),
        (date(2026, 12, 31), "2026-12-28", "2027-01-03"),
        (date(2027, 1, 3), "2026-12-28", "2027-01-03"),
    ],
)
def test_week_resource_on_month_and_year_boundary(today, start, end):
    week = build_week(parse_schedule(SCHEDULE_DATA), today, "Europe/Moscow")
    assert (week.week_start, week.week_end) == (start, end)
    assert [day.weekday for day in week.days][0] == "понедельник"
    assert len(week.days) == 7


def test_week_uses_configured_timezone_not_utc_date():
    # 31.12.2026 22:30 UTC = 01.01.2027 01:30 в Москве — неделя та же, но «сегодня» уже пятница.
    deps = make_deps(clock=lambda: datetime(2026, 12, 31, 22, 30, tzinfo=UTC))
    week = current_week(deps)
    assert week.week_start == "2026-12-28"
    assert week.days[4].date == "2027-01-01"


async def test_resource_is_discoverable_and_readable_via_mcp():
    server = build_server(make_deps())
    async with create_connected_server_and_client_session(server) as session:
        resources = await session.list_resources()
        content = await session.read_resource(AnyUrl(WEEK_RESOURCE_URI))
    assert [str(item.uri) for item in resources.resources] == [WEEK_RESOURCE_URI]
    assert '"week_start":"2026-10-12"' in content.contents[0].text
    assert content.contents[0].mimeType == "application/json"


# --- Discovery и схемы --------------------------------------------------------------


async def test_tools_discovery_exposes_strict_input_and_output_schemas():
    server = build_server(make_deps())
    async with create_connected_server_and_client_session(server) as session:
        listed = await session.list_tools()
        result = await session.call_tool("get_schedule", {"date": "2026-10-12"})
    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == {"get_weather", "get_schedule", "add_reminder"}
    for tool in tools.values():
        assert tool.inputSchema["additionalProperties"] is False
        assert tool.outputSchema is not None
    assert tools["get_weather"].inputSchema["properties"]["city"]["maxLength"] == 100
    assert tools["add_reminder"].inputSchema["required"] == ["text", "remind_at"]
    assert tools["add_reminder"].annotations.readOnlyHint is False
    assert tools["get_schedule"].annotations.readOnlyHint is True
    assert isinstance(result, types.CallToolResult)
    assert result.structuredContent["date"] == "2026-10-12"


async def test_extra_arguments_are_rejected_by_server_schema():
    async with running_toolbox(make_deps()) as toolbox:
        with pytest.raises(MCPToolError) as error:
            await toolbox.call_tool("get_weather", {"city": "Казань", "chat_id": 42})
    assert error.value.code == "invalid_arguments"


# --- Напоминания на сервере ---------------------------------------------------------


async def test_add_reminder_without_trusted_owner_is_refused():
    # Arrange: независимый клиент вызывает инструмент без owner_token.
    store = MemoryReminderStore()
    async with running_toolbox(make_deps(store=store)) as toolbox:
        # Act
        with pytest.raises(MCPToolError) as missing:
            await toolbox.call_tool("add_reminder", {"text": "тест", "remind_at": FUTURE})
        with pytest.raises(MCPToolError) as forged:
            await toolbox.call_tool(
                "add_reminder",
                {"text": "тест", "remind_at": FUTURE},
                meta={META_KEY: "forged.token"},
            )
    # Assert
    assert missing.value.code == "owner_missing"
    assert forged.value.code == "owner_invalid"
    assert store.rows == {}


async def test_owner_in_model_arguments_is_rejected():
    store = MemoryReminderStore()
    async with running_toolbox(make_deps(store=store)) as toolbox:
        with pytest.raises(MCPToolError) as error:
            await toolbox.call_tool(
                "add_reminder",
                {"text": "тест", "remind_at": FUTURE, "chat_id": 999},
                meta=owner_meta(),
            )
    assert error.value.code == "invalid_arguments"
    assert store.rows == {}


async def test_server_without_secret_refuses_even_signed_token():
    async with running_toolbox(make_deps(secret="")) as toolbox:
        with pytest.raises(MCPToolError) as error:
            await toolbox.call_tool(
                "add_reminder", {"text": "тест", "remind_at": FUTURE}, meta=owner_meta()
            )
    assert error.value.code == "owner_invalid"


async def test_future_reminder_is_created_with_normalized_time():
    store = MemoryReminderStore()
    async with running_toolbox(make_deps(store=store)) as toolbox:
        result = await toolbox.call_tool(
            "add_reminder",
            {"text": " отправить отчёт ", "remind_at": "2026-10-13T15:30:00Z"},
            meta=owner_meta(chat_id=7),
        )
    assert result == {
        "reminder_id": 1,
        "text": "отправить отчёт",
        "remind_at": "2026-10-13T18:30:00+03:00",
        "timezone": "Europe/Moscow",
        "created": True,
    }
    assert store.rows["action-1"]["chat_id"] == 7


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"text": "отчёт", "remind_at": "2026-10-12T08:59:00+03:00"}, "remind_at_in_past"),
        ({"text": "отчёт", "remind_at": "2026-10-13T18:30:00"}, "remind_at_no_timezone"),
        ({"text": "   ", "remind_at": FUTURE}, "invalid_arguments"),
        ({"text": "", "remind_at": FUTURE}, "invalid_arguments"),
        ({"text": "x" * 501, "remind_at": FUTURE}, "invalid_arguments"),
    ],
)
async def test_past_time_or_empty_text_is_rejected(arguments, code):
    store = MemoryReminderStore()
    async with running_toolbox(make_deps(store=store)) as toolbox:
        with pytest.raises(MCPToolError) as error:
            await toolbox.call_tool("add_reminder", arguments, meta=owner_meta())
    assert error.value.code == code
    assert store.rows == {}


async def test_same_idempotency_key_does_not_create_second_record():
    # Arrange
    store = MemoryReminderStore()
    owner = owner_from_meta(OWNER_SECRET, owner_meta(chat_id=5, action_id="same"))
    args = ReminderInput(text="сделать перерыв", remind_at=FUTURE)
    # Act
    first = await add_reminder(args, owner, store, now=FIXED_NOW)
    second = await add_reminder(args, owner, store, now=FIXED_NOW)
    # Assert
    assert (first.created, second.created) == (True, False)
    assert first.reminder_id == second.reminder_id
    assert len(store.rows) == 1


async def test_other_user_cannot_reuse_action_key():
    store = MemoryReminderStore()
    args = ReminderInput(text="тест", remind_at=FUTURE)
    first_owner = owner_from_meta(OWNER_SECRET, owner_meta(1, "k"))
    await add_reminder(args, first_owner, store, now=FIXED_NOW)
    with pytest.raises(ToolError) as error:
        await add_reminder(
            args, owner_from_meta(OWNER_SECRET, owner_meta(2, "k")), store, now=FIXED_NOW
        )
    assert error.value.code == "owner_invalid"


def test_expired_owner_token_is_rejected():
    token = sign_owner_token(OWNER_SECRET, OwnerContext(1, "a", "Europe/Moscow"), ttl_seconds=-1)
    with pytest.raises(ToolError) as error:
        owner_from_meta(OWNER_SECRET, {META_KEY: token})
    assert error.value.code == "owner_invalid"
