"""Единое представление ошибок MCP и внешних сервисов для пользователя."""

from __future__ import annotations


class MCPError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


class MCPUnavailableError(MCPError):
    """Сервер не запущен, разорвал соединение или не ответил вовремя."""


class MCPToolError(MCPError):
    """Инструмент вернул isError с кодом ошибки."""


class MCPBadResultError(MCPError):
    """Результат не соответствует ожидаемой структуре."""


USER_MESSAGES = {
    "mcp_unavailable": "Инструменты сейчас недоступны: MCP-сервер не подключён. Попробуйте позже.",
    "mcp_timeout": "Инструмент не ответил вовремя. Попробуйте позже.",
    "unknown_tool": "Агент выбрал недоступное действие, запрос отклонён. Переформулируйте его.",
    "invalid_arguments": "Параметры действия не прошли проверку. Уточните запрос.",
    "bad_result": "Сервис вернул данные неожиданной структуры. Попробуйте позже.",
    "weather_timeout": "Погодный сервис не ответил вовремя. Попробуйте позже.",
    "weather_unavailable": "Погодный сервис временно недоступен. Попробуйте позже.",
    "weather_bad_response": "Погодный сервис вернул некорректный ответ. Попробуйте позже.",
    "schedule_unavailable": "Расписание сейчас недоступно. Попробуйте позже.",
    "storage_unavailable": "Хранилище напоминаний недоступно.",
    "owner_missing": "Не удалось подтвердить владельца действия.",
    "owner_invalid": "Не удалось подтвердить владельца действия.",
    "remind_at_in_past": "Время напоминания уже прошло.",
    "remind_at_no_timezone": "У времени напоминания нет часового пояса.",
    "limit_exceeded": (
        "Запрос требует больше двух обращений к инструментам. Разбейте его на части."
    ),
}


def user_message(code: str) -> str:
    return USER_MESSAGES.get(code, "Не удалось выполнить действие. Попробуйте позже.")
