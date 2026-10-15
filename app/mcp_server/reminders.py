"""Создание напоминаний: проверка доверенного владельца, времени и идемпотентность."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from app.config import is_valid_timezone
from app.mcp_server.errors import ToolError
from app.mcp_server.schemas import ReminderInput, ReminderOutput
from app.owner_token import OwnerContext, OwnerTokenError, verify_owner_token


class ReminderStore(Protocol):
    async def insert_once(
        self, *, chat_id: int, text: str, remind_at: datetime, timezone: str, key: str
    ) -> tuple[int, bool, dict[str, Any]]: ...


class PostgresReminderStore:
    """INSERT ... ON CONFLICT по ключу идемпотентности: повтор возвращает прежнюю запись."""

    def __init__(self, pool_factory: Callable[[], Awaitable[Any]]) -> None:
        self._pool_factory = pool_factory

    async def insert_once(
        self, *, chat_id: int, text: str, remind_at: datetime, timezone: str, key: str
    ) -> tuple[int, bool, dict[str, Any]]:
        pool = await self._pool_factory()
        row = await pool.fetchrow(
            """
            INSERT INTO reminders (chat_id, text, remind_at, timezone, idempotency_key)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id, chat_id, text, remind_at, timezone
            """,
            chat_id,
            text,
            remind_at,
            timezone,
            key,
        )
        created = row is not None
        if row is None:
            row = await pool.fetchrow(
                """
                SELECT id, chat_id, text, remind_at, timezone
                FROM reminders WHERE idempotency_key = $1
                """,
                key,
            )
        if row is None or row["chat_id"] != chat_id:
            raise ToolError("owner_invalid", "Действие принадлежит другому владельцу.")
        return int(row["id"]), created, dict(row)


def owner_from_meta(secret: str, meta: Any) -> OwnerContext:
    token = getattr(meta, "owner_token", None) if meta is not None else None
    if token is None and isinstance(meta, dict):
        token = meta.get("owner_token")
    if token is None:
        raise ToolError(
            "owner_missing",
            "Нет доверенного контекста владельца: создание напоминаний доступно только боту.",
        )
    try:
        owner = verify_owner_token(secret, token)
    except OwnerTokenError:
        raise ToolError("owner_invalid", "Доверенный контекст владельца недействителен.") from None
    if not is_valid_timezone(owner.timezone):
        raise ToolError("owner_invalid", "Часовой пояс владельца недействителен.")
    return owner


def parse_remind_at(raw: str, tz_name: str, now: datetime) -> datetime:
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        raise ToolError(
            "invalid_arguments", "remind_at должен быть датой и временем ISO 8601."
        ) from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ToolError(
            "remind_at_no_timezone", "remind_at должен содержать явное смещение часового пояса."
        )
    if moment <= now:
        raise ToolError("remind_at_in_past", "Время напоминания уже прошло.")
    return moment.astimezone(ZoneInfo(tz_name))


async def add_reminder(
    arguments: ReminderInput,
    owner: OwnerContext,
    store: ReminderStore,
    *,
    now: datetime | None = None,
) -> ReminderOutput:
    current = now or datetime.now(UTC)
    remind_at = parse_remind_at(arguments.remind_at, owner.timezone, current)
    reminder_id, created, row = await store.insert_once(
        chat_id=owner.chat_id,
        text=arguments.text,
        remind_at=remind_at,
        timezone=owner.timezone,
        key=owner.action_id,
    )
    stored_at = row.get("remind_at", remind_at)
    if isinstance(stored_at, datetime):
        stored_at = stored_at.astimezone(ZoneInfo(row.get("timezone", owner.timezone)))
    return ReminderOutput(
        reminder_id=reminder_id,
        text=row.get("text", arguments.text),
        remind_at=stored_at.isoformat(),
        timezone=row.get("timezone", owner.timezone),
        created=created,
    )
