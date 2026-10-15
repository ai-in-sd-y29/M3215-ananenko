"""Запуск MCP-сервера через stdio: python -m app.mcp_server [--env-file .env]."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import asyncpg
from mcp.server.stdio import stdio_server

from app.config import ConfigError
from app.mcp_server.config import ServerSettings
from app.mcp_server.errors import ToolError
from app.mcp_server.reminders import PostgresReminderStore
from app.mcp_server.schedule import load_schedule
from app.mcp_server.server import ServerDeps, build_server
from app.mcp_server.weather import HttpJsonFetcher

logger = logging.getLogger("app.mcp_server")


class LazyPool:
    """Пул PostgreSQL создаётся только при первом вызове add_reminder."""

    def __init__(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> asyncpg.Pool:
        async with self._lock:
            if self._pool is None:
                try:
                    self._pool = await asyncpg.create_pool(
                        host=self._settings.postgres_host,
                        port=self._settings.postgres_port,
                        database=self._settings.postgres_db,
                        user=self._settings.postgres_user,
                        password=self._settings.postgres_password,
                        min_size=1,
                        max_size=2,
                        timeout=5,
                        command_timeout=5,
                    )
                except (OSError, asyncpg.PostgresError, TimeoutError):
                    logger.warning("PostgreSQL недоступен для add_reminder")
                    raise ToolError(
                        "storage_unavailable", "Хранилище напоминаний недоступно."
                    ) from None
            return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()


async def serve(settings: ServerSettings) -> None:
    fetcher = HttpJsonFetcher(timeout_seconds=settings.weather_timeout_seconds)
    pool = LazyPool(settings)
    deps = ServerDeps(
        fetcher=fetcher,
        load_schedule=lambda: load_schedule(settings.schedule_path),
        reminder_store=PostgresReminderStore(pool.get),
        owner_secret=settings.owner_secret,
        schedule_timezone=settings.schedule_timezone,
    )
    if not settings.owner_secret:
        logger.info("MCP_OWNER_SECRET не задан: add_reminder будет отклонять все вызовы")
    server = build_server(deps)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await fetcher.aclose()
        await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP-сервер ассистента студента (stdio)")
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args()
    # stdout занят протоколом MCP, журнал пишется только в stderr.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        settings = ServerSettings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    asyncio.run(serve(settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
