"""Регистрация инструментов и ресурса в низкоуровневом MCP-сервере."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from pydantic import AnyUrl, BaseModel, ValidationError

from app.mcp_server.errors import ToolError
from app.mcp_server.reminders import ReminderStore, add_reminder, owner_from_meta
from app.mcp_server.schedule import Schedule, build_week, parse_day, schedule_for_day
from app.mcp_server.schemas import (
    ReminderInput,
    ReminderOutput,
    ScheduleInput,
    ScheduleOutput,
    ToolErrorDetail,
    ToolErrorPayload,
    WeatherInput,
    WeatherOutput,
    WeekOutput,
    input_schema,
    output_schema,
)
from app.mcp_server.weather import JsonFetcher, get_weather

logger = logging.getLogger("app.mcp_server")

SERVER_NAME = "student-assistant"
WEEK_RESOURCE_URI = "schedule://current-week"


@dataclass
class ServerDeps:
    fetcher: JsonFetcher
    load_schedule: Callable[[], Schedule]
    reminder_store: ReminderStore
    owner_secret: str
    schedule_timezone: str
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


TOOLS: list[types.Tool] = [
    types.Tool(
        name="get_weather",
        title="Текущая погода",
        description=(
            "Текущая погода в населённом пункте: температура °C, описание, ветер, "
            "время наблюдения. Используй для любых вопросов о погоде сейчас или сегодня "
            "(температура, осадки, зонт, ветер). Если городов несколько, вернёт варианты."
        ),
        inputSchema=input_schema(WeatherInput),
        outputSchema=output_schema(WeatherOutput),
        annotations=types.ToolAnnotations(
            readOnlyHint=True, idempotentHint=True, openWorldHint=True
        ),
    ),
    types.Tool(
        name="get_schedule",
        title="Расписание на дату",
        description=(
            "Занятия студента на одну дату: время начала и конца, предмет, аудитория. "
            "Используй для вопросов о парах, занятиях и расписании на конкретный день. "
            "Пустой список означает, что занятий нет."
        ),
        inputSchema=input_schema(ScheduleInput),
        outputSchema=output_schema(ScheduleOutput),
        annotations=types.ToolAnnotations(
            readOnlyHint=True, idempotentHint=True, openWorldHint=False
        ),
    ),
    types.Tool(
        name="add_reminder",
        title="Создание напоминания",
        description=(
            "Создаёт напоминание пользователю на конкретные дату и время. Вызывай, когда "
            "пользователь просит напомнить и указал точную дату или относительное время "
            "(«завтра в 18:30», «через два часа»). Выполняется только после подтверждения "
            "пользователем."
        ),
        inputSchema=input_schema(ReminderInput),
        outputSchema=output_schema(ReminderOutput),
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
]


def _error_result(code: str, message: str) -> types.CallToolResult:
    payload = ToolErrorPayload(error=ToolErrorDetail(code=code, message=message))
    data = payload.model_dump()
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
        structuredContent=data,
        isError=True,
    )


def _validate(model: type[BaseModel], arguments: dict[str, Any]) -> Any:
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]) for error in exc.errors() if error["loc"])
        raise ToolError("invalid_arguments", f"Аргументы не прошли проверку: {fields}.") from None


def build_server(deps: ServerDeps) -> Server:
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        try:
            result = await _dispatch(server, deps, name, arguments)
        except ToolError as exc:
            logger.info("tool=%s status=error code=%s", name, exc.code)
            return _error_result(exc.code, exc.message)
        logger.info("tool=%s status=ok", name)
        return result.model_dump(mode="json")

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl(WEEK_RESOURCE_URI),
                name="current-week",
                title="Расписание текущей недели",
                description="Занятия с понедельника по воскресенье, сгруппированные по датам.",
                mimeType="application/json",
            )
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        if str(uri) != WEEK_RESOURCE_URI:
            raise ValueError("Неизвестный ресурс")
        week = current_week(deps)
        return [ReadResourceContents(content=week.model_dump_json(), mime_type="application/json")]

    return server


def current_week(deps: ServerDeps) -> WeekOutput:
    today = deps.clock().astimezone(ZoneInfo(deps.schedule_timezone)).date()
    return build_week(deps.load_schedule(), today, deps.schedule_timezone)


async def _dispatch(
    server: Server, deps: ServerDeps, name: str, arguments: dict[str, Any]
) -> BaseModel:
    if name == "get_weather":
        args = _validate(WeatherInput, arguments)
        return await get_weather(args.city, deps.fetcher)
    if name == "get_schedule":
        args = _validate(ScheduleInput, arguments)
        day = parse_day(args.date)
        return schedule_for_day(deps.load_schedule(), day, deps.schedule_timezone)
    if name == "add_reminder":
        meta = server.request_context.meta
        owner = owner_from_meta(deps.owner_secret, meta)
        args = _validate(ReminderInput, arguments)
        return await add_reminder(args, owner, deps.reminder_store, now=deps.clock())
    raise ToolError("unknown_tool", "Инструмент не найден.")
