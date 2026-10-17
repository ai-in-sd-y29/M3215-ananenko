from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
from asyncpg import Pool

from app.agent.loop import Agent, AgentContext, confirmation_text
from app.agent.mcp_client import MCPToolbox
from app.config import Settings
from app.llm.client import (
    LLMClient,
    LLMEmptyResponseError,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.services.audit import AuditRecord, AuditService
from app.services.history import HistoryService
from app.services.messaging import prepare_telegram_parts, strip_markdown
from app.services.pending import PendingActionService
from app.services.users import UserService

router = Router(name="chat")

USER_ERROR_TIMEOUT = (
    "Не удалось получить ответ: сервис модели не ответил вовремя. Попробуйте позже."
)
USER_ERROR_UNAVAILABLE = (
    "Не удалось получить ответ: сервис модели временно недоступен. Попробуйте позже."
)
USER_ERROR_EMPTY = "Модель вернула пустой ответ. Переформулируйте запрос или попробуйте снова."
USER_ERROR_GENERIC = "Не удалось обработать запрос. Попробуйте позже."
KNOWN_COMMANDS = "/start /study /translate /exam /settings /reset /tools /timezone /why /week"


def confirm_keyboard(action_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Подтвердить", callback_data=f"rem:ok:{action_id}"),
                InlineKeyboardButton(text="Отменить", callback_data=f"rem:no:{action_id}"),
            ]
        ]
    )


async def _send_answer_parts(status: Message, message: Message, answer: str) -> None:
    parts = prepare_telegram_parts(answer)
    if not parts:
        await status.edit_text(USER_ERROR_EMPTY, parse_mode=None)
        return
    first_text, first_mode = parts[0]
    try:
        await status.edit_text(first_text, parse_mode=first_mode)
    except TelegramBadRequest:
        await status.edit_text(strip_markdown(answer)[:4096], parse_mode=None)
        return
    for part_text, part_mode in parts[1:]:
        try:
            await message.answer(part_text, parse_mode=part_mode)
        except TelegramBadRequest:
            await message.answer(strip_markdown(part_text), parse_mode=None)


LLM_ERRORS: list[tuple[type[LLMError], str, str]] = [
    (LLMTimeoutError, USER_ERROR_TIMEOUT, "llm_timeout"),
    (LLMEmptyResponseError, USER_ERROR_EMPTY, "llm_empty"),
    (LLMUnavailableError, USER_ERROR_UNAVAILABLE, "llm_unavailable"),
    (LLMError, USER_ERROR_GENERIC, "llm_error"),
]


@router.message(F.chat.type == "private", F.text)
async def handle_private_text(
    message: Message,
    db: Pool,
    settings: Settings,
    llm: LLMClient,
    mcp: MCPToolbox,
) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    if text.startswith("/"):
        # Неизвестные команды не отправляем в модель.
        await message.answer(f"Неизвестная команда. Доступны: {KNOWN_COMMANDS}", parse_mode=None)
        return

    chat_id = message.chat.id
    pending = PendingActionService(db)
    source_key = f"{chat_id}:{message.message_id}"
    existing = await pending.get_by_source(chat_id, source_key)
    if existing is not None:
        # Повторная доставка того же update: вторую операцию не готовим.
        if existing.status == "pending":
            await message.answer(
                confirmation_text(existing.arguments, existing.timezone),
                reply_markup=confirm_keyboard(existing.id),
                parse_mode=None,
            )
        return

    users = UserService(db)
    history = HistoryService(
        db,
        max_messages=settings.history_max_messages,
        max_chars=settings.history_max_chars,
    )
    audit = AuditService(db)
    user = await users.get_or_create(chat_id)
    context = await history.context_for_llm(chat_id, text)
    ctx = AgentContext(
        chat_id=chat_id,
        source_key=source_key,
        user_text=text,
        mode=user.mode,
        temperature=user.temperature,
        timezone=user.timezone,
        history=context,
    )
    agent = Agent(llm, mcp, pending, schedule_timezone=settings.schedule_timezone)

    status = await message.answer("Готовлю ответ…", parse_mode=None)
    await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
    await history.add_message(chat_id, "user", text)

    try:
        result = await agent.run(ctx)
    except LLMError as exc:
        for error_type, user_text, code in LLM_ERRORS:
            if isinstance(exc, error_type):
                await audit.record(
                    AuditRecord(
                        chat_id=chat_id,
                        request_id=ctx.request_id,
                        action="error",
                        validation_status="не выполнялась",
                        execution_status=f"ошибка: {code}",
                        reason="сервис модели не вернул решение",
                    )
                )
                await status.edit_text(user_text, parse_mode=None)
                return
        raise

    await audit.record(result.audit)
    await history.add_message(chat_id, "assistant", result.text)
    if result.kind == "confirm" and result.pending is not None:
        # Отдельное сообщение с inline-кнопками надёжнее edit «Готовлю ответ…»:
        # у части клиентов Telegram markup после edit не показывается.
        keyboard = confirm_keyboard(result.pending.id)
        try:
            await status.delete()
        except TelegramBadRequest:
            try:
                await status.edit_text(result.text, parse_mode=None)
            except TelegramBadRequest:
                pass
        await message.answer(result.text, reply_markup=keyboard, parse_mode=None)
        return
    await _send_answer_parts(status, message, result.text)
