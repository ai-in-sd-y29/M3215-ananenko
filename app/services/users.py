from dataclasses import dataclass

import asyncpg

from app.config import ALLOWED_TEMPERATURES, DEFAULT_MODE, MODES, is_valid_timezone


@dataclass(frozen=True)
class UserSettings:
    chat_id: int
    mode: str
    temperature: float
    timezone: str | None = None


def parse_temperature(raw: str) -> float | None:
    try:
        value = float(raw.strip().replace(",", "."))
    except ValueError:
        return None
    for allowed in ALLOWED_TEMPERATURES:
        if abs(value - allowed) < 1e-9:
            return allowed
    return None


class UserService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_or_create(self, chat_id: int) -> UserSettings:
        row = await self._pool.fetchrow(
            "SELECT chat_id, mode, temperature, timezone FROM user_settings WHERE chat_id = $1",
            chat_id,
        )
        if row is None:
            await self._pool.execute(
                """
                INSERT INTO user_settings (chat_id, mode, temperature)
                VALUES ($1, $2, $3)
                ON CONFLICT (chat_id) DO NOTHING
                """,
                chat_id,
                DEFAULT_MODE,
                0.7,
            )
            return UserSettings(chat_id=chat_id, mode=DEFAULT_MODE, temperature=0.7)
        timezone = row["timezone"]
        return UserSettings(
            chat_id=row["chat_id"],
            mode=row["mode"] if row["mode"] in MODES else DEFAULT_MODE,
            temperature=float(row["temperature"]),
            timezone=timezone if timezone and is_valid_timezone(timezone) else None,
        )

    async def set_mode(self, chat_id: int, mode: str) -> UserSettings:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        await self.get_or_create(chat_id)
        await self._pool.execute(
            """
            UPDATE user_settings
            SET mode = $2, updated_at = NOW()
            WHERE chat_id = $1
            """,
            chat_id,
            mode,
        )
        return await self.get_or_create(chat_id)

    async def set_temperature(self, chat_id: int, temperature: float) -> UserSettings:
        if temperature not in ALLOWED_TEMPERATURES:
            raise ValueError("invalid temperature")
        await self.get_or_create(chat_id)
        await self._pool.execute(
            """
            UPDATE user_settings
            SET temperature = $2, updated_at = NOW()
            WHERE chat_id = $1
            """,
            chat_id,
            temperature,
        )
        return await self.get_or_create(chat_id)

    async def set_timezone(self, chat_id: int, timezone: str) -> UserSettings:
        if not is_valid_timezone(timezone):
            raise ValueError("unknown timezone")
        await self.get_or_create(chat_id)
        await self._pool.execute(
            """
            UPDATE user_settings
            SET timezone = $2, updated_at = NOW()
            WHERE chat_id = $1
            """,
            chat_id,
            timezone,
        )
        return await self.get_or_create(chat_id)
