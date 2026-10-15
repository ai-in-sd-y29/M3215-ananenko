"""Краткий аудит решений агента; источник данных для /why."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import asyncpg

logger = logging.getLogger("app.audit")


@dataclass(frozen=True)
class AuditRecord:
    chat_id: int
    request_id: str
    action: str
    validation_status: str
    execution_status: str
    tool_name: str | None = None
    args_summary: str | None = None
    reason: str | None = None
    mcp_calls: int = 0
    duration_ms: int = 0
    created_at: datetime | None = None


class AuditService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record(self, record: AuditRecord) -> None:
        # В текстовый журнал попадают только request_id и статусы, без chat_id и текста.
        logger.info(
            "agent request_id=%s action=%s tool=%s validation=%s execution=%s "
            "mcp_calls=%s duration_ms=%s",
            record.request_id,
            record.action,
            record.tool_name or "-",
            record.validation_status,
            record.execution_status,
            record.mcp_calls,
            record.duration_ms,
        )
        await self._pool.execute(
            """
            INSERT INTO agent_audit (
                chat_id, request_id, action, tool_name, args_summary,
                validation_status, execution_status, reason, mcp_calls, duration_ms
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            record.chat_id,
            record.request_id,
            record.action,
            record.tool_name,
            record.args_summary,
            record.validation_status,
            record.execution_status,
            record.reason,
            record.mcp_calls,
            record.duration_ms,
        )

    async def latest(self, chat_id: int) -> AuditRecord | None:
        row = await self._pool.fetchrow(
            """
            SELECT chat_id, request_id, created_at, action, tool_name, args_summary,
                   validation_status, execution_status, reason, mcp_calls, duration_ms
            FROM agent_audit
            WHERE chat_id = $1
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            chat_id,
        )
        if row is None:
            return None
        return AuditRecord(
            chat_id=row["chat_id"],
            request_id=row["request_id"],
            action=row["action"],
            validation_status=row["validation_status"],
            execution_status=row["execution_status"],
            tool_name=row["tool_name"],
            args_summary=row["args_summary"],
            reason=row["reason"],
            mcp_calls=row["mcp_calls"],
            duration_ms=row["duration_ms"],
            created_at=row["created_at"],
        )
