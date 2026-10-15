from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from asyncpg import Pool

from app.agent.errors import MCPError, user_message
from app.agent.mcp_client import MCPToolbox
from app.config import is_valid_timezone
from app.services.audit import AuditRecord, AuditService
from app.services.timezones import offset_label
from app.services.users import UserService

router = Router(name="agent_commands")

WEEK_RESOURCE_URI = "schedule://current-week"
TOOLS_UNAVAILABLE = "Инструменты сейчас недоступны: MCP-сервер не подключён. Попробуйте позже."
TIMEZONE_HINT = (
    "Укажите часовой пояс в формате IANA, например: /timezone Europe/Moscow, "
    "/timezone Asia/Yekaterinburg. Смещения вида +03:00 не подходят."
)
ACTION_TITLES = {
    "respond": "ответ без инструмента",
    "clarify": "уточнение",
    "call_tool": "вызов инструмента",
    "rejected": "отклонено (недопустимое действие)",
    "limit": "остановка: превышен лимит вызовов",
    "error": "ошибка сервиса модели",
}


def format_tools(mcp: MCPToolbox) -> str:
    tools = mcp.tools
    if not mcp.available or not tools:
        return TOOLS_UNAVAILABLE
    lines = ["Доступные инструменты (получены от MCP-сервера):"]
    for tool in tools:
        kind = "чтение" if tool.read_only else "изменение, нужно подтверждение"
        description = tool.description.split(". ")[0].rstrip(".")
        lines.append(f"• {tool.name} — {description} ({kind})")
    resources = mcp.resources
    if resources:
        lines.append("")
        lines.append("Ресурсы:")
        lines.extend(f"• {item.uri} — {item.description or item.name}" for item in resources)
    return "\n".join(lines)


def format_why(record: AuditRecord | None, tz_name: str | None) -> str:
    if record is None:
        return "Вы ещё не отправляли запросов агенту."
    zone = ZoneInfo(tz_name) if tz_name else UTC
    created = (record.created_at or datetime.now(UTC)).astimezone(zone)
    action = ACTION_TITLES.get(record.action, record.action)
    if record.tool_name and record.action in {"call_tool", "rejected", "limit"}:
        action = f"{action} {record.tool_name}"
    lines = [
        f"Последний запрос: {created:%d.%m.%Y %H:%M} ({tz_name or 'UTC'})",
        f"Действие: {action}",
    ]
    if record.args_summary:
        lines.append(f"Аргументы: {record.args_summary}")
    lines.extend(
        [
            f"Проверка: {record.validation_status}",
            f"Выполнение: {record.execution_status}",
            f"Вызовов MCP: {record.mcp_calls}, длительность: {record.duration_ms} мс",
            f"Причина: {record.reason or '—'}",
        ]
    )
    return "\n".join(lines)


def format_week(data: dict) -> str:
    lines = [
        f"Расписание на неделю {data['week_start']} — {data['week_end']} ({data['timezone']}):"
    ]
    for day in data["days"]:
        lines.append("")
        lines.append(f"{day['weekday'].capitalize()}, {day['date']}:")
        if not day["lessons"]:
            lines.append("  занятий нет")
        for lesson in day["lessons"]:
            lines.append(
                f"  {lesson['start']}–{lesson['end']} {lesson['title']}, {lesson['location']}"
            )
    return "\n".join(lines)


@router.message(Command("tools"), F.chat.type == "private")
async def cmd_tools(message: Message, mcp: MCPToolbox) -> None:
    await message.answer(format_tools(mcp), parse_mode=None)


@router.message(Command("timezone"), F.chat.type == "private")
async def cmd_timezone(message: Message, command: CommandObject, db: Pool) -> None:
    users = UserService(db)
    argument = (command.args or "").strip()
    if not argument:
        user = await users.get_or_create(message.chat.id)
        if user.timezone is None:
            await message.answer(f"Часовой пояс не задан.\n{TIMEZONE_HINT}", parse_mode=None)
            return
        now = datetime.now(ZoneInfo(user.timezone))
        await message.answer(
            f"Текущий часовой пояс: {user.timezone} ({offset_label(now)}). "
            f"Местное время: {now:%d.%m.%Y %H:%M}.",
            parse_mode=None,
        )
        return
    if not is_valid_timezone(argument):
        await message.answer(f"Неизвестный часовой пояс.\n{TIMEZONE_HINT}", parse_mode=None)
        return
    user = await users.set_timezone(message.chat.id, argument)
    now = datetime.now(ZoneInfo(argument))
    await message.answer(
        f"Часовой пояс сохранён: {argument} ({offset_label(now)}). "
        f"Местное время: {now:%d.%m.%Y %H:%M}.",
        parse_mode=None,
    )


@router.message(Command("why"), F.chat.type == "private")
async def cmd_why(message: Message, db: Pool) -> None:
    user = await UserService(db).get_or_create(message.chat.id)
    record = await AuditService(db).latest(message.chat.id)
    await message.answer(format_why(record, user.timezone), parse_mode=None)


@router.message(Command("week"), F.chat.type == "private")
async def cmd_week(message: Message, mcp: MCPToolbox) -> None:
    try:
        data = await mcp.read_resource(WEEK_RESOURCE_URI)
        text = format_week(data)
    except MCPError as exc:
        text = user_message(exc.code) if exc.code != "mcp_unavailable" else TOOLS_UNAVAILABLE
    except (KeyError, TypeError):
        text = user_message("bad_result")
    await message.answer(text, parse_mode=None)
