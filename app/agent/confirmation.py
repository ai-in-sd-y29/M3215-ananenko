"""Подтверждение и отмена подготовленного напоминания."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Protocol

from app.agent.errors import MCPError, user_message
from app.agent.loop import REASONS, summarize_arguments
from app.owner_token import META_KEY, OwnerContext, sign_owner_token
from app.services.audit import AuditRecord
from app.services.pending import ClaimResult
from app.services.timezones import offset_label

logger = logging.getLogger("app.agent.confirmation")


class PendingRepo(Protocol):
    async def claim(self, chat_id: int, action_id: str) -> ClaimResult: ...

    async def complete(self, action_id: str, result: dict[str, Any]) -> None: ...

    async def fail(self, action_id: str, code: str) -> None: ...

    async def cancel(self, chat_id: int, action_id: str) -> ClaimResult: ...


class AuditRepo(Protocol):
    async def record(self, record: AuditRecord) -> None: ...


class ToolCaller(Protocol):
    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


def _valid_reminder_result(data: dict[str, Any]) -> bool:
    try:
        datetime.fromisoformat(data["remind_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        isinstance(data.get("reminder_id"), int)
        and isinstance(data.get("timezone"), str)
        and isinstance(data.get("created"), bool)
    )


def created_text(result: dict[str, Any], *, repeated: bool) -> str:
    from app.services.formatting import format_human_datetime

    moment = datetime.fromisoformat(result["remind_at"])
    when = f"{format_human_datetime(moment)} ({result['timezone']}, {offset_label(moment)})"
    if repeated or not result.get("created", True):
        return (
            f"<b>Напоминание №{result['reminder_id']}</b> уже создано: "
            f"<i>{when}</i>. Повтор не выполнялся."
        )
    return f"<b>Напоминание №{result['reminder_id']}</b> создано: <i>{when}</i>."


STATUS_TEXTS = {
    "not_found": "Действие не найдено или принадлежит другому пользователю.",
    "expired": "Подтверждение истекло (действует 5 минут). Повторите запрос.",
    "cancelled": "Это действие уже отменено.",
    "executing": "Напоминание уже создаётся, подождите.",
    "failed": "Это действие завершилось ошибкой. Повторите запрос.",
}


class ReminderConfirmation:
    def __init__(
        self,
        pending: PendingRepo,
        audit: AuditRepo,
        toolbox: ToolCaller,
        owner_secret: str,
    ) -> None:
        self._pending = pending
        self._audit = audit
        self._toolbox = toolbox
        self._secret = owner_secret

    async def _record(self, chat_id: int, request_id: str, started: float, **fields: Any) -> None:
        await self._audit.record(
            AuditRecord(
                chat_id=chat_id,
                request_id=request_id,
                duration_ms=int((time.perf_counter() - started) * 1000),
                **fields,
            )
        )

    async def confirm(self, chat_id: int, action_id: str, request_id: str | None = None) -> str:
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        claim = await self._pending.claim(chat_id, action_id)
        if claim.kind == "done" and claim.action and claim.action.result:
            return created_text(claim.action.result, repeated=True)
        if claim.kind != "claimed" or claim.action is None:
            if claim.kind in {"expired", "cancelled"} and claim.action is not None:
                await self._record(
                    chat_id,
                    request_id,
                    started,
                    action="call_tool",
                    tool_name=claim.action.tool_name,
                    args_summary=summarize_arguments(
                        claim.action.tool_name, claim.action.arguments
                    ),
                    validation_status=f"отклонена: {claim.kind}",
                    execution_status="не выполнялось",
                    reason="подтверждение недействительно",
                )
            return STATUS_TEXTS.get(claim.kind, STATUS_TEXTS["not_found"])

        action = claim.action
        summary = summarize_arguments(action.tool_name, action.arguments)
        token = sign_owner_token(
            self._secret,
            OwnerContext(chat_id=chat_id, action_id=action.id, timezone=action.timezone),
        )
        try:
            data = await self._toolbox.call_tool(
                action.tool_name, action.arguments, meta={META_KEY: token}
            )
            if not _valid_reminder_result(data):
                raise MCPError("bad_result", "unexpected reminder result")
        except MCPError as exc:
            await self._pending.fail(action.id, exc.code)
            await self._record(
                chat_id,
                request_id,
                started,
                action="call_tool",
                tool_name=action.tool_name,
                args_summary=summary,
                validation_status="пройдена",
                execution_status=f"ошибка: {exc.code}",
                reason=REASONS.get(action.tool_name, ""),
                mcp_calls=1,
            )
            logger.info("reminder failed request_id=%s code=%s", request_id, exc.code)
            return f"Напоминание не создано. {user_message(exc.code)}"

        await self._pending.complete(action.id, data)
        await self._record(
            chat_id,
            request_id,
            started,
            action="call_tool",
            tool_name=action.tool_name,
            args_summary=summary,
            validation_status="пройдена, подтверждено пользователем",
            execution_status="успешно" if data["created"] else "успешно (повтор, без дубликата)",
            reason=REASONS.get(action.tool_name, ""),
            mcp_calls=1,
        )
        return created_text(data, repeated=False)

    async def cancel(self, chat_id: int, action_id: str, request_id: str | None = None) -> str:
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        result = await self._pending.cancel(chat_id, action_id)
        if result.kind == "claimed" and result.action is not None:
            await self._record(
                chat_id,
                request_id,
                started,
                action="call_tool",
                tool_name=result.action.tool_name,
                args_summary=summarize_arguments(result.action.tool_name, result.action.arguments),
                validation_status="пройдена",
                execution_status="отменено пользователем",
                reason="пользователь отменил действие",
            )
            return "Создание напоминания отменено."
        if result.kind == "done" and result.action and result.action.result:
            return created_text(result.action.result, repeated=True)
        return STATUS_TEXTS.get(result.kind, STATUS_TEXTS["not_found"])
