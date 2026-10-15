"""Lab 2 на реальном PostgreSQL: RUN_INTEGRATION=1 python -m pytest -m integration -v."""

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest

from app.agent.confirmation import ReminderConfirmation
from app.agent.errors import MCPToolError
from app.db import create_pool, init_schema
from app.mcp_server.reminders import PostgresReminderStore
from app.owner_token import META_KEY, OwnerContext, sign_owner_token
from app.services.audit import AuditRecord, AuditService
from app.services.pending import PendingActionService
from app.services.users import UserService
from tests.lab2_support import OWNER_SECRET, make_deps, running_toolbox

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]


def future_arguments() -> dict[str, str]:
    moment = (datetime.now(UTC) + timedelta(days=1)).replace(microsecond=0)
    return {"text": "отправить отчёт", "remind_at": moment.isoformat()}


async def test_repeated_update_and_concurrent_confirm_create_one_reminder(database):
    # Arrange
    settings, *_ = database
    pool = await create_pool(settings)
    await init_schema(pool)

    async def pool_factory():
        return pool

    pending = PendingActionService(pool)
    store = PostgresReminderStore(pool_factory)
    deps = make_deps(store=store, clock=lambda: datetime.now(UTC))
    try:
        # Act 1: один и тот же update обработан пять раз параллельно.
        prepared = await asyncio.gather(
            *[
                pending.create(
                    chat_id=1,
                    source_key="1:10",
                    tool_name="add_reminder",
                    arguments=future_arguments(),
                    timezone="Europe/Moscow",
                )
                for _ in range(5)
            ]
        )
        # Assert 1: одна подготовленная операция.
        assert len({action.id for action, _ in prepared}) == 1
        assert sum(created for _, created in prepared) == 1
        assert await pool.fetchval("SELECT count(*) FROM pending_actions") == 1
        action_id = prepared[0][0].id

        async with running_toolbox(deps) as toolbox:
            confirmation = ReminderConfirmation(pending, AuditService(pool), toolbox, OWNER_SECRET)
            # Act 2: пользователь чужого чата пытается подтвердить и отменить.
            foreign_confirm = await confirmation.confirm(2, action_id)
            foreign_cancel = await confirmation.cancel(2, action_id)
            # Act 3: двойное конкурентное подтверждение владельцем.
            texts = await asyncio.gather(*[confirmation.confirm(1, action_id) for _ in range(5)])
            # Act 4: сетевой повтор MCP-вызова с тем же ключом идемпотентности.
            token = sign_owner_token(OWNER_SECRET, OwnerContext(1, action_id, "Europe/Moscow"))
            replay = await toolbox.call_tool(
                "add_reminder", prepared[0][0].arguments, meta={META_KEY: token}
            )
            # Act 5: другой владелец с тем же ключом.
            foreign_token = sign_owner_token(
                OWNER_SECRET, OwnerContext(2, action_id, "Europe/Moscow")
            )
            with pytest.raises(MCPToolError) as foreign_replay:
                await toolbox.call_tool(
                    "add_reminder", prepared[0][0].arguments, meta={META_KEY: foreign_token}
                )

        # Assert 2–5
        assert "другому пользователю" in foreign_confirm
        assert "другому пользователю" in foreign_cancel
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
        created = [text for text in texts if "создано:" in text and "уже" not in text]
        assert len(created) == 1
        reminder = await pool.fetchrow("SELECT id, chat_id, text FROM reminders")
        assert (reminder["chat_id"], reminder["text"]) == (1, "отправить отчёт")
        assert replay["created"] is False
        assert replay["reminder_id"] == reminder["id"]
        assert foreign_replay.value.code == "owner_invalid"
        status = await pool.fetchval("SELECT status FROM pending_actions WHERE id = $1", action_id)
        assert status == "done"
        again = await ReminderConfirmation(
            pending, AuditService(pool), toolbox, OWNER_SECRET
        ).confirm(1, action_id)
        assert "уже создано" in again
    finally:
        await pool.close()


async def test_expiry_new_action_and_user_isolation_in_postgres(database):
    # Arrange
    settings, *_ = database
    pool = await create_pool(settings)
    await init_schema(pool)
    try:
        expired_service = PendingActionService(pool, ttl_seconds=0)
        service = PendingActionService(pool)
        expired, _ = await expired_service.create(
            chat_id=1,
            source_key="1:1",
            tool_name="add_reminder",
            arguments=future_arguments(),
            timezone="Europe/Moscow",
        )
        first, _ = await service.create(
            chat_id=3,
            source_key="3:1",
            tool_name="add_reminder",
            arguments=future_arguments(),
            timezone="Europe/Moscow",
        )
        changed = future_arguments() | {"text": "другой текст"}
        second, _ = await service.create(
            chat_id=3,
            source_key="3:2",
            tool_name="add_reminder",
            arguments=changed,
            timezone="Europe/Moscow",
        )
        # Act
        expired_claim = await service.claim(1, expired.id)
        old_claim = await service.claim(3, first.id)
        foreign_claim = await service.claim(1, second.id)
        await UserService(pool).set_timezone(3, "Asia/Tokyo")
        audit = AuditService(pool)
        for chat_id, action in ((3, "respond"), (4, "call_tool")):
            await audit.record(
                AuditRecord(
                    chat_id=chat_id,
                    request_id="r",
                    action=action,
                    validation_status="ok",
                    execution_status="ok",
                )
            )
        # Assert
        assert expired_claim.kind == "expired"
        assert old_claim.kind == "cancelled"  # изменённое действие требует нового подтверждения
        assert foreign_claim.kind == "not_found"
        assert (await service.claim(3, second.id)).kind == "claimed"
        assert (await UserService(pool).get_or_create(3)).timezone == "Asia/Tokyo"
        assert (await audit.latest(3)).action == "respond"
        assert (await audit.latest(4)).action == "call_tool"
        assert await audit.latest(5) is None
    finally:
        await pool.close()
