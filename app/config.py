"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import available_timezones

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


ALLOWED_TEMPERATURES = (0.0, 0.3, 0.7, 1.0)
DEFAULT_MODE = "study"
MODES = ("study", "translate", "exam")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_PATH = str(PROJECT_ROOT / "data" / "schedule.json")


def is_valid_timezone(name: str) -> bool:
    """IANA-зона из базы tzdata; смещения вида +03:00 и пути не принимаются."""
    return bool(name) and name in _available_timezones()


@cache
def _available_timezones() -> frozenset[str]:
    return frozenset(available_timezones())


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    llm_timeout_seconds: float = 60.0
    history_max_messages: int = 20
    history_max_chars: int = 12_000
    schedule_path: str = DEFAULT_SCHEDULE_PATH
    schedule_timezone: str = "Europe/Moscow"
    weather_timeout_seconds: float = 8.0
    mcp_call_timeout_seconds: float = 25.0

    @classmethod
    def load(
        cls, env_file: Path | str = ".env", *, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def positive_int(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if result < 1:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно целое число не меньше 1.") from None

        def positive_float(key: str, default: str) -> float:
            try:
                result = float(value(key, default))
                if result <= 0:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно положительное число.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        openrouter_api_key = value("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ConfigError("OPENROUTER_API_KEY: укажите ключ доступа OpenRouter.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        base_url = value("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError("OPENROUTER_BASE_URL: нужен http:// или https:// URL.")
        model = value("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        if not model:
            raise ConfigError("OPENROUTER_MODEL: укажите имя модели OpenRouter.")
        schedule_timezone = value("SCHEDULE_TIMEZONE", "Europe/Moscow")
        if not is_valid_timezone(schedule_timezone):
            raise ConfigError("SCHEDULE_TIMEZONE: нужна зона IANA, например Europe/Moscow.")
        schedule_path = Path(value("SCHEDULE_PATH", DEFAULT_SCHEDULE_PATH))
        if not schedule_path.is_absolute():
            schedule_path = PROJECT_ROOT / schedule_path
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=base_url,
            openrouter_model=model,
            llm_timeout_seconds=positive_float("LLM_TIMEOUT_SECONDS", "60"),
            history_max_messages=positive_int("HISTORY_MAX_MESSAGES", "20"),
            history_max_chars=positive_int("HISTORY_MAX_CHARS", "12000"),
            schedule_path=str(schedule_path),
            schedule_timezone=schedule_timezone,
            weather_timeout_seconds=positive_float("WEATHER_TIMEOUT_SECONDS", "8"),
            mcp_call_timeout_seconds=positive_float("MCP_CALL_TIMEOUT_SECONDS", "25"),
        )
