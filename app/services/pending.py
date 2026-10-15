"""Подготовленные действия с побочным эффектом и их подтверждение."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import asyncpg

CONFIRMATION_TTL_SECONDS = 300

ClaimKind = Literal["claimed", "not_found", "expired", "cancelled", "executing", "done", "failed"]


@dataclass(frozen=True)
class PendingAction:
    id: str
    chat_id: int
    tool_name: str
    arguments: dict[str, Any]
    timezone: str
    status: str
    result: dict[str, Any] | None
    expires_at: datetime


@dataclass(frozen=True)
class ClaimResult:
    kind: ClaimKind
    action: PendingAction | None = None


def _json(value: Any) -> Any:
    if value is None or isinstance(value, dict):
        return value
    return json.loads(value)


def _action(row: Any) -> PendingAction:
    return PendingAction(
        id=row["id"],
        chat_id=row["chat_id"],
        tool_name=row["tool_name"],
        arguments=_json(row["arguments"]),
        timezone=row["timezone"],
        status=row["status"],
        result=_json(row["result"]),
        expires_at=row["expires_at"],
    )


COLUMNS = "id, chat_id, tool_name, arguments, timezone, status, result, expires_at"


class PendingActionService:
    def __init__(self, pool: asyncpg.Pool, *, ttl_seconds: int = CONFIRMATION_TTL_SECONDS) -> None:
        self._pool = pool
        self._ttl = ttl_seconds

    async def get_by_source(self, chat_id: int, source_key: str) -> PendingAction | None:
        row = await self._pool.fetchrow(
            f"SELECT {COLUMNS} FROM pending_actions WHERE source_key = $1 AND chat_id = $2",
            source_key,
            chat_id,
        )
        return _action(row) if row else None

    async def create(
        self,
        *,
        chat_id: int,
        source_key: str,
        tool_name: str,
        arguments: dict[str, Any],
        timezone: str,
    ) -> tuple[PendingAction, bool]:
        """Одна подготовленная операция на исходное сообщение; прежние ожидающие отменяются."""
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"""
                INSERT INTO pending_actions
                    (id, chat_id, source_key, tool_name, arguments, timezone, status, expires_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'pending',
                        NOW() + make_interval(secs => $7))
                ON CONFLICT (source_key) DO NOTHING
                RETURNING {COLUMNS}
                """,
                uuid.uuid4().hex,
                chat_id,
                source_key,
                tool_name,
                json.dumps(arguments, ensure_ascii=False),
                timezone,
                float(self._ttl),
            )
            if row is None:
                existing = await conn.fetchrow(
                    f"SELECT {COLUMNS} FROM pending_actions WHERE source_key = $1", source_key
                )
                if existing is None or existing["chat_id"] != chat_id:
                    raise PermissionError("source key belongs to another chat")
                return _action(existing), False
            await conn.execute(
                """
                UPDATE pending_actions SET status = 'cancelled', updated_at = NOW()
                WHERE chat_id = $1 AND status = 'pending' AND id <> $2
                """,
                chat_id,
                row["id"],
            )
            return _action(row), True

    async def claim(self, chat_id: int, action_id: str) -> ClaimResult:
        """Атомарно переводит pending → executing; выигрывает ровно одно подтверждение."""
        row = await self._pool.fetchrow(
            f"""
            UPDATE pending_actions SET status = 'executing', updated_at = NOW()
            WHERE id = $1 AND chat_id = $2 AND status = 'pending' AND expires_at > NOW()
            RETURNING {COLUMNS}
            """,
            action_id,
            chat_id,
        )
        if row is not None:
            return ClaimResult("claimed", _action(row))
        row = await self._pool.fetchrow(
            f"SELECT {COLUMNS} FROM pending_actions WHERE id = $1 AND chat_id = $2",
            action_id,
            chat_id,
        )
        if row is None:
            return ClaimResult("not_found")
        action = _action(row)
        if action.status == "pending":
            await self._pool.execute(
                """
                UPDATE pending_actions SET status = 'expired', updated_at = NOW()
                WHERE id = $1 AND status = 'pending'
                """,
                action_id,
            )
            return ClaimResult("expired", action)
        return ClaimResult(action.status, action)  # type: ignore[arg-type]

    async def complete(self, action_id: str, result: dict[str, Any]) -> None:
        await self._pool.execute(
            """
            UPDATE pending_actions SET status = 'done', result = $2::jsonb, updated_at = NOW()
            WHERE id = $1 AND status = 'executing'
            """,
            action_id,
            json.dumps(result, ensure_ascii=False),
        )

    async def fail(self, action_id: str, code: str) -> None:
        await self._pool.execute(
            """
            UPDATE pending_actions SET status = 'failed', result = $2::jsonb, updated_at = NOW()
            WHERE id = $1 AND status = 'executing'
            """,
            action_id,
            json.dumps({"error": code}),
        )

    async def cancel(self, chat_id: int, action_id: str) -> ClaimResult:
        row = await self._pool.fetchrow(
            f"""
            UPDATE pending_actions SET status = 'cancelled', updated_at = NOW()
            WHERE id = $1 AND chat_id = $2 AND status = 'pending'
            RETURNING {COLUMNS}
            """,
            action_id,
            chat_id,
        )
        if row is not None:
            # claimed: именно этот вызов выполнил переход pending → cancelled.
            return ClaimResult("claimed", _action(row))
        row = await self._pool.fetchrow(
            f"SELECT {COLUMNS} FROM pending_actions WHERE id = $1 AND chat_id = $2",
            action_id,
            chat_id,
        )
        if row is None:
            return ClaimResult("not_found")
        action = _action(row)
        return ClaimResult(action.status, action)  # type: ignore[arg-type]
