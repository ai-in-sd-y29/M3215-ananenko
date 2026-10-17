import re

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery
from asyncpg import Pool

from app.agent.confirmation import ReminderConfirmation
from app.agent.mcp_client import MCPToolbox
from app.services.audit import AuditService
from app.services.pending import PendingActionService

router = Router(name="reminders")

CALLBACK_PATTERN = re.compile(r"^rem:(ok|no):([0-9a-f]{32})$")


@router.callback_query(F.data.startswith("rem:"))
async def on_reminder_callback(
    callback: CallbackQuery, db: Pool, mcp: MCPToolbox, owner_secret: str
) -> None:
    message = callback.message
    match = CALLBACK_PATTERN.match(callback.data or "")
    if message is None or message.chat.type != "private" or match is None:
        await callback.answer("Некорректная кнопка", show_alert=True)
        return
    # Владелец берётся из доверенного update: личный чат и тот же пользователь.
    if callback.from_user is None or callback.from_user.id != message.chat.id:
        await callback.answer("Действие недоступно", show_alert=True)
        return
    verb, action_id = match.groups()
    service = ReminderConfirmation(PendingActionService(db), AuditService(db), mcp, owner_secret)
    if verb == "ok":
        text = await service.confirm(message.chat.id, action_id)
    else:
        text = await service.cancel(message.chat.id, action_id)
    try:
        await message.edit_text(text, reply_markup=None, parse_mode="HTML")
    except TelegramBadRequest:
        try:
            await message.edit_text(text, reply_markup=None, parse_mode=None)
        except TelegramBadRequest:
            try:
                await message.answer(text, parse_mode="HTML")
            except TelegramBadRequest:
                await message.answer(text, parse_mode=None)
    await callback.answer()
