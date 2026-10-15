"""Настройки MCP-сервера: только то, что нужно инструментам, без токенов бота и LLM."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from app.config import DEFAULT_SCHEDULE_PATH, PROJECT_ROOT, ConfigError, is_valid_timezone

OWNER_SECRET_ENV = "MCP_OWNER_SECRET"


@dataclass(frozen=True)
class ServerSettings:
    schedule_path: str = DEFAULT_SCHEDULE_PATH
    schedule_timezone: str = "Europe/Moscow"
    weather_timeout_seconds: float = 8.0
    owner_secret: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    postgres_password: str = field(default="", repr=False)

    @classmethod
    def load(
        cls, env_file: Path | str | None = None, *, environ: Mapping[str, str] | None = None
    ) -> ServerSettings:
        values = {
            **(dotenv_values(env_file, interpolate=False) if env_file else {}),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        tz_name = value("SCHEDULE_TIMEZONE", "Europe/Moscow")
        if not is_valid_timezone(tz_name):
            raise ConfigError("SCHEDULE_TIMEZONE: нужна зона IANA, например Europe/Moscow.")
        schedule_path = Path(value("SCHEDULE_PATH", DEFAULT_SCHEDULE_PATH))
        if not schedule_path.is_absolute():
            schedule_path = PROJECT_ROOT / schedule_path
        try:
            timeout = float(value("WEATHER_TIMEOUT_SECONDS", "8"))
            port = int(value("POSTGRES_PORT", "5432"))
            if timeout <= 0 or not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            raise ConfigError("WEATHER_TIMEOUT_SECONDS/POSTGRES_PORT: неверное число.") from None
        return cls(
            schedule_path=str(schedule_path),
            schedule_timezone=tz_name,
            weather_timeout_seconds=timeout,
            # Секрет задаёт только бот для своего дочернего процесса; .env его не содержит.
            owner_secret=(os.environ if environ is None else environ).get(OWNER_SECRET_ENV, ""),
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port,
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            postgres_password=value("POSTGRES_PASSWORD"),
        )
