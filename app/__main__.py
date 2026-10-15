import argparse
import asyncio
import logging
import secrets

from aiogram import Dispatcher

from app.agent.mcp_client import MCPToolbox, server_parameters, stdio_connector
from app.config import ConfigError, Settings
from app.db import create_pool, init_schema
from app.handlers import router
from app.health import HealthState, start_health_server
from app.llm.client import LLMClient
from app.logging_setup import configure_logging
from app.telegram import create_bot

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    llm = LLMClient(settings)
    # Секрет доверенного контекста знает только бот и его дочерний MCP-сервер.
    owner_secret = secrets.token_urlsafe(32)
    mcp = MCPToolbox(
        stdio_connector(server_parameters(settings, owner_secret)),
        call_timeout_seconds=settings.mcp_call_timeout_seconds,
    )
    runner = None
    try:
        state.pool = await create_pool(settings)
        await init_schema(state.pool)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен, схема готова.")
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        await mcp.start()
        if not mcp.available:
            logger.warning("Бот работает без инструментов: MCP-сервер недоступен.")
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        dispatcher = Dispatcher()
        dispatcher.include_router(router)
        runner = await start_health_server(state, settings.health_port)
        state.polling_task = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                db=state.pool,
                settings=settings,
                llm=llm,
                mcp=mcp,
                owner_secret=owner_secret,
                allowed_updates=dispatcher.resolve_used_update_types(),
                close_bot_session=False,
            )
        )
        state.initialized = True
        await state.polling_task
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await mcp.stop()
        await llm.aclose()
        await bot.session.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-ассистент студента в Telegram")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен и прокси.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
