from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from asyncpg import Pool

from app.config import ALLOWED_TEMPERATURES, Settings
from app.llm.prompts import mode_title
from app.services.formatting import start_message_html
from app.services.history import HistoryService
from app.services.users import UserService, parse_temperature

router = Router(name="commands")


def _settings_keyboard() -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton(text=str(value), callback_data=f"temp:{value}")
        for value in ALLOWED_TEMPERATURES
    ]
    return InlineKeyboardMarkup(inline_keyboard=[row])


async def _answer_html(message: Message, text: str) -> None:
    try:
        await message.answer(text, parse_mode="HTML")
    except TelegramBadRequest:
        await message.answer(text, parse_mode=None)


async def _switch_mode(message: Message, db: Pool, mode: str) -> None:
    users = UserService(db)
    history = HistoryService(db, max_messages=1, max_chars=1)
    await history.clear(message.chat.id)
    await users.set_mode(message.chat.id, mode)
    await _answer_html(
        message,
        f"<b>Режим переключён:</b> {mode_title(mode)}.\n<i>История предыдущего режима очищена.</i>",
    )


@router.message(CommandStart(), F.chat.type == "private")
async def cmd_start(message: Message) -> None:
    await _answer_html(message, start_message_html())


@router.message(Command("study"), F.chat.type == "private")
async def cmd_study(message: Message, db: Pool) -> None:
    await _switch_mode(message, db, "study")


@router.message(Command("translate"), F.chat.type == "private")
async def cmd_translate(message: Message, db: Pool) -> None:
    await _switch_mode(message, db, "translate")


@router.message(Command("exam"), F.chat.type == "private")
async def cmd_exam(message: Message, db: Pool) -> None:
    await _switch_mode(message, db, "exam")


@router.message(Command("reset"), F.chat.type == "private")
async def cmd_reset(message: Message, db: Pool, settings: Settings) -> None:
    history = HistoryService(
        db,
        max_messages=settings.history_max_messages,
        max_chars=settings.history_max_chars,
    )
    await history.clear(message.chat.id)
    await _answer_html(
        message,
        "<b>История диалога очищена.</b>\n<i>Режим и temperature сохранены.</i>",
    )


@router.message(Command("settings"), F.chat.type == "private")
async def cmd_settings(
    message: Message,
    command: CommandObject,
    db: Pool,
    settings: Settings,
) -> None:
    users = UserService(db)
    user = await users.get_or_create(message.chat.id)
    argument = (command.args or "").strip()
    if argument:
        temperature = parse_temperature(argument)
        if temperature is None:
            await _answer_html(
                message,
                "<b>Некорректное значение temperature.</b>\n<i>Допустимы: 0.0, 0.3, 0.7, 1.0.</i>",
            )
            return
        user = await users.set_temperature(message.chat.id, temperature)
        await _answer_html(
            message,
            f"<b>Temperature обновлён:</b> <code>{user.temperature}</code>.\n"
            f"<i>Применится со следующего запроса.</i>",
        )
        return

    text = (
        f"<b>Текущие настройки:</b>\n"
        f"режим: <b>{mode_title(user.mode)}</b>\n"
        f"модель: <code>{settings.openrouter_model}</code>\n"
        f"temperature: <code>{user.temperature}</code>\n\n"
        "<i>Выберите temperature кнопкой или командой</i>\n"
        "<code>/settings 0.0|0.3|0.7|1.0</code>"
    )
    try:
        await message.answer(text, reply_markup=_settings_keyboard(), parse_mode="HTML")
    except TelegramBadRequest:
        await message.answer(text, reply_markup=_settings_keyboard(), parse_mode=None)


@router.callback_query(F.data.startswith("temp:"))
async def on_temperature_callback(callback: CallbackQuery, db: Pool) -> None:
    if callback.message is None or callback.message.chat.type != "private":
        await callback.answer()
        return
    raw = (callback.data or "").removeprefix("temp:")
    temperature = parse_temperature(raw)
    if temperature is None:
        await callback.answer("Некорректное значение", show_alert=True)
        return
    users = UserService(db)
    user = await users.set_temperature(callback.message.chat.id, temperature)
    try:
        await callback.message.answer(
            f"<b>Temperature обновлён:</b> <code>{user.temperature}</code>.\n"
            f"<i>Применится со следующего запроса.</i>",
            parse_mode="HTML",
        )
    except TelegramBadRequest:
        await callback.message.answer(
            f"Temperature обновлён: {user.temperature}. Применится со следующего запроса.",
            parse_mode=None,
        )
    await callback.answer()
