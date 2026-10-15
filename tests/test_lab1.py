"""Обязательные сценарии лабораторной работы № 1."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.filters import CommandObject

from app.config import ConfigError, Settings
from app.handlers.chat import USER_ERROR_UNAVAILABLE, handle_private_text
from app.handlers.commands import (
    cmd_reset,
    cmd_settings,
    cmd_study,
    on_temperature_callback,
)
from app.llm.client import ChatResult, LLMUnavailableError
from app.llm.prompts import STUDY_PROMPT, system_prompt_for
from app.services.history import HistoryMessage, HistoryService, trim_history
from app.services.messaging import (
    markdown_to_telegram_html,
    prepare_telegram_parts,
    split_telegram_text,
)
from app.services.users import UserSettings, parse_temperature

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
OPENROUTER_KEY = "test-openrouter-key"


def make_settings(**overrides) -> Settings:
    values = {
        "bot_token": TOKEN,
        "postgres_password": "secret-db",
        "openrouter_api_key": OPENROUTER_KEY,
        "openrouter_model": "openai/gpt-4o-mini",
        "history_max_messages": 4,
        "history_max_chars": 40,
    }
    values.update(overrides)
    return Settings(**values)


def private_message(*, chat_id: int, text: str) -> MagicMock:
    message = MagicMock()
    message.chat.id = chat_id
    message.chat.type = "private"
    message.message_id = 100
    message.text = text
    message.answer = AsyncMock()
    message.bot = MagicMock()
    message.bot.send_chat_action = AsyncMock()
    return message


def no_tools() -> MagicMock:
    toolbox = MagicMock()
    toolbox.tools = []
    toolbox.spec = MagicMock(return_value=None)
    return toolbox


def agent_services():
    pending = AsyncMock()
    pending.get_by_source = AsyncMock(return_value=None)
    return (
        patch("app.handlers.chat.PendingActionService", return_value=pending),
        patch("app.handlers.chat.AuditService", return_value=AsyncMock()),
    )


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD=secret-db\nOPENROUTER_API_KEY={OPENROUTER_KEY}\n",
        encoding="utf-8",
    )
    return Settings.load(path, environ={})


def test_settings_require_openrouter_key(tmp_path):
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings.load(
            tmp_path / ".env",
            environ={"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "secret"},
        )


def test_message_uses_active_mode_system_prompt():
    # Arrange / Act
    prompt = system_prompt_for("study")
    # Assert
    assert prompt == STUDY_PROMPT
    assert "программирования" in prompt


def test_trim_history_keeps_order_and_current_request_budget():
    # Arrange
    history = [
        HistoryMessage("user", "aaaa"),
        HistoryMessage("assistant", "bbbb"),
        HistoryMessage("user", "cccc"),
        HistoryMessage("assistant", "dddd"),
        HistoryMessage("user", "eeee"),
    ]
    # Act: max 3 messages, 20 chars total, reserve 8 for current request → budget 12
    trimmed = trim_history(history, max_messages=3, max_chars=20, reserved_chars=8)
    # Assert
    assert [item.content for item in trimmed] == ["cccc", "dddd", "eeee"]
    assert len("".join(item.content for item in trimmed)) <= 12


async def test_histories_of_two_users_do_not_mix():
    # Arrange
    store: dict[int, list[HistoryMessage]] = {
        1: [HistoryMessage("user", "привет от 1"), HistoryMessage("assistant", "ответ 1")],
        2: [HistoryMessage("user", "привет от 2"), HistoryMessage("assistant", "ответ 2")],
    }

    class FakePool:
        async def fetch(self, query, *args):
            chat_id = args[0]
            return [{"role": m.role, "content": m.content} for m in store.get(chat_id, [])]

        async def execute(self, query, *args):
            return None

    service = HistoryService(FakePool(), max_messages=10, max_chars=1000)

    # Act
    first = await service.list_messages(1)
    second = await service.list_messages(2)
    # Assert
    assert [m.content for m in first] == ["привет от 1", "ответ 1"]
    assert [m.content for m in second] == ["привет от 2", "ответ 2"]


async def test_mode_switch_clears_history_and_saves_mode():
    # Arrange
    message = private_message(chat_id=7, text="/study")
    db = object()
    users = AsyncMock()
    users.set_mode = AsyncMock(return_value=UserSettings(chat_id=7, mode="study", temperature=0.7))
    history = AsyncMock()
    history.clear = AsyncMock()
    with (
        patch("app.handlers.commands.UserService", return_value=users),
        patch("app.handlers.commands.HistoryService", return_value=history),
    ):
        # Act
        await cmd_study(message, db)  # type: ignore[arg-type]
    # Assert
    history.clear.assert_awaited_once_with(7)
    users.set_mode.assert_awaited_once_with(7, "study")
    assert "study" in message.answer.await_args.args[0]


async def test_reset_clears_only_caller_history():
    # Arrange
    message = private_message(chat_id=11, text="/reset")
    cleared: list[int] = []

    class FakeHistory:
        def __init__(self, *args, **kwargs):
            pass

        async def clear(self, chat_id: int) -> None:
            cleared.append(chat_id)

    with patch("app.handlers.commands.HistoryService", FakeHistory):
        # Act
        await cmd_reset(message, object(), make_settings())  # type: ignore[arg-type]
    # Assert
    assert cleared == [11]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.0", 0.0), ("0.3", 0.3), ("0.7", 0.7), ("1.0", 1.0), ("2", None), ("abc", None)],
)
def test_settings_temperature_validation(raw, expected):
    assert parse_temperature(raw) == expected


async def test_settings_accepts_valid_and_rejects_invalid():
    # Arrange
    db = object()
    settings = make_settings()
    users = AsyncMock()
    users.get_or_create = AsyncMock(
        return_value=UserSettings(chat_id=5, mode="study", temperature=0.7)
    )
    users.set_temperature = AsyncMock(
        return_value=UserSettings(chat_id=5, mode="study", temperature=0.3)
    )
    with patch("app.handlers.commands.UserService", return_value=users):
        ok = private_message(chat_id=5, text="/settings 0.3")
        bad = private_message(chat_id=5, text="/settings 2")
        # Act
        await cmd_settings(
            ok,
            CommandObject(prefix="/", command="settings", args="0.3"),
            db,
            settings,
        )  # type: ignore[arg-type]
        await cmd_settings(
            bad,
            CommandObject(prefix="/", command="settings", args="2"),
            db,
            settings,
        )  # type: ignore[arg-type]
    # Assert
    users.set_temperature.assert_awaited_once_with(5, 0.3)
    assert "0.3" in ok.answer.await_args.args[0]
    assert "Некорректное" in bad.answer.await_args.args[0]


async def test_llm_error_becomes_safe_user_message():
    # Arrange
    message = private_message(chat_id=3, text="Объясни список")
    status = MagicMock()
    status.edit_text = AsyncMock()
    message.answer = AsyncMock(return_value=status)

    users = AsyncMock()
    users.get_or_create = AsyncMock(
        return_value=UserSettings(chat_id=3, mode="study", temperature=0.7)
    )
    history = AsyncMock()
    history.context_for_llm = AsyncMock(return_value=[])
    history.add_message = AsyncMock()
    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=LLMUnavailableError("down"))
    pending_patch, audit_patch = agent_services()

    with (
        patch("app.handlers.chat.UserService", return_value=users),
        patch("app.handlers.chat.HistoryService", return_value=history),
        pending_patch,
        audit_patch,
    ):
        # Act
        await handle_private_text(message, object(), make_settings(), llm, no_tools())  # type: ignore[arg-type]
    # Assert
    status.edit_text.assert_awaited_once()
    assert status.edit_text.await_args.args[0] == USER_ERROR_UNAVAILABLE
    assert "traceback" not in status.edit_text.await_args.args[0].lower()
    assert history.add_message.await_count == 1
    assert history.add_message.await_args.args[1] == "user"


def test_long_answer_is_split_without_loss():
    # Arrange
    text = "абвг" * 2000  # 8000 символов
    # Act
    parts = split_telegram_text(text, limit=4096)
    # Assert
    assert all(len(part) <= 4096 for part in parts)
    assert "".join(parts) == text
    assert len(parts) == 2


def test_markdown_converted_to_telegram_html():
    # Arrange
    raw = "**Пояснение:**\n1. **Классы** — шаблоны\n```python\nclass Dog:\n    pass\n```"
    # Act
    html_text = markdown_to_telegram_html(raw)
    # Assert
    assert "<b>Пояснение:</b>" in html_text
    assert "<b>Классы</b>" in html_text
    assert "<pre>" in html_text
    assert "class Dog:" in html_text
    assert "**" not in html_text
    assert "```" not in html_text
    parts = prepare_telegram_parts(raw)
    assert parts == [(html_text, "HTML")]


async def test_private_text_calls_llm_with_active_mode_prompt():
    # Arrange
    message = private_message(chat_id=9, text="Что такое список?")
    status_message = MagicMock()
    status_message.edit_text = AsyncMock()
    message.answer = AsyncMock(return_value=status_message)

    users = AsyncMock()
    users.get_or_create = AsyncMock(
        return_value=UserSettings(chat_id=9, mode="study", temperature=0.3)
    )
    history = AsyncMock()
    history.context_for_llm = AsyncMock(
        return_value=[
            {"role": "user", "content": "раньше"},
            {"role": "assistant", "content": "ответ"},
        ]
    )
    history.add_message = AsyncMock()
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=ChatResult(content="Краткое объяснение"))
    pending_patch, audit_patch = agent_services()

    with (
        patch("app.handlers.chat.UserService", return_value=users),
        patch("app.handlers.chat.HistoryService", return_value=history),
        pending_patch,
        audit_patch,
    ):
        # Act
        await handle_private_text(message, object(), make_settings(), llm, no_tools())  # type: ignore[arg-type]

    # Assert
    messages = llm.chat.await_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(STUDY_PROMPT)
    assert llm.chat.await_args.kwargs["temperature"] == 0.3
    assert messages[-1] == {"role": "user", "content": "Что такое список?"}
    assert messages[1]["role"] == "user"
    status_message.edit_text.assert_awaited_once()
    assert status_message.edit_text.await_args.args[0] == markdown_to_telegram_html(
        "Краткое объяснение"
    )
    assert status_message.edit_text.await_args.kwargs["parse_mode"] == "HTML"


async def test_temperature_callback_saves_value():
    # Arrange
    message = private_message(chat_id=4, text="/settings")
    callback = MagicMock()
    callback.data = "temp:1.0"
    callback.message = message
    callback.answer = AsyncMock()
    users = AsyncMock()
    users.set_temperature = AsyncMock(
        return_value=UserSettings(chat_id=4, mode="study", temperature=1.0)
    )
    with patch("app.handlers.commands.UserService", return_value=users):
        # Act
        await on_temperature_callback(callback, object())  # type: ignore[arg-type]
    # Assert
    users.set_temperature.assert_awaited_once_with(4, 1.0)


async def test_dispatcher_sends_split_parts_in_order():
    # Arrange: ответ длиннее 4096 — уходит как plain text без HTML
    long_text = "x" * 5000
    settings = make_settings()
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=ChatResult(content=long_text))
    users = AsyncMock()
    users.get_or_create = AsyncMock(
        return_value=UserSettings(chat_id=42, mode="exam", temperature=0.7)
    )
    history = AsyncMock()
    history.context_for_llm = AsyncMock(return_value=[])
    history.add_message = AsyncMock()

    message = private_message(chat_id=42, text="Тема: циклы")
    sent: list[tuple[str, str | None]] = []

    async def fake_answer(text, **kwargs):
        sent.append((text, kwargs.get("parse_mode")))
        status = MagicMock()

        async def edit_text(new_text, **kw):
            sent[0] = (new_text, kw.get("parse_mode"))

        status.edit_text = edit_text
        return status

    message.answer = fake_answer
    pending_patch, audit_patch = agent_services()

    with (
        patch("app.handlers.chat.UserService", return_value=users),
        patch("app.handlers.chat.HistoryService", return_value=history),
        pending_patch,
        audit_patch,
    ):
        await handle_private_text(message, object(), settings, llm, no_tools())  # type: ignore[arg-type]

    assert sent[0][0] == long_text[:4096]
    assert sent[1][0] == long_text[4096:]
    assert "".join(part for part, _ in sent) == long_text
