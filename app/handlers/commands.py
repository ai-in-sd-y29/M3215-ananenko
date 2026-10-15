from aiogram import F, Router
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
from app.services.history import HistoryService
from app.services.users import UserService, parse_temperature

router = Router(name="commands")


def _settings_keyboard() -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton(text=str(value), callback_data=f"temp:{value}")
        for value in ALLOWED_TEMPERATURES
    ]
    return InlineKeyboardMarkup(inline_keyboard=[row])


async def _switch_mode(message: Message, db: Pool, mode: str) -> None:
    users = UserService(db)
    history = HistoryService(db, max_messages=1, max_chars=1)
    await history.clear(message.chat.id)
    await users.set_mode(message.chat.id, mode)
    await message.answer(
        f"Режим переключён: {mode_title(mode)}. История предыдущего режима очищена.",
        parse_mode=None,
    )


@router.message(CommandStart(), F.chat.type == "private")
async def cmd_start(message: Message) -> None:
    text = (
        "Я AI-ассистент студента.\n\n"
        "Команды:\n"
        "/study — объяснение учебных вопросов по программированию\n"
        "/translate — перевод текста на указанный язык\n"
        "/exam — экзаменатор: вопросы по теме и проверка ответов\n"
        "/settings — режим, модель и temperature\n"
        "/reset — очистить историю диалога\n"
        "/tools — какие действия агент может выполнить через MCP-сервер\n"
        "/timezone <зона> — часовой пояс для напоминаний, например Europe/Moscow\n"
        "/why — почему агент выбрал действие в последнем запросе\n"
        "/week — расписание на текущую неделю\n\n"
        "Отправьте текстовое сообщение — отвечу в активном режиме "
        "(по умолчанию /study). Могу узнать погоду, найти занятия на дату и "
        "подготовить напоминание (создам его только после вашего подтверждения)."
    )
    await message.answer(text, parse_mode=None)


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
    await message.answer(
        "История диалога очищена. Режим и temperature сохранены.",
        parse_mode=None,
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
            await message.answer(
                "Некорректное значение temperature. Допустимы: 0.0, 0.3, 0.7, 1.0.",
                parse_mode=None,
            )
            return
        user = await users.set_temperature(message.chat.id, temperature)
        await message.answer(
            f"Temperature обновлён: {user.temperature}. Применится со следующего запроса.",
            parse_mode=None,
        )
        return

    text = (
        f"Текущие настройки:\n"
        f"режим: {mode_title(user.mode)}\n"
        f"модель: {settings.openrouter_model}\n"
        f"temperature: {user.temperature}\n\n"
        "Выберите temperature кнопкой или командой "
        "/settings 0.0|0.3|0.7|1.0"
    )
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
    await callback.message.answer(
        f"Temperature обновлён: {user.temperature}. Применится со следующего запроса.",
        parse_mode=None,
    )
    await callback.answer()
