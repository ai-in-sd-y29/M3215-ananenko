"""Клиент OpenRouter (OpenAI-совместимый Chat Completions)."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from app.config import Settings

logger = logging.getLogger("app.llm")


class LLMError(Exception):
    """Базовая ошибка обращения к языковой модели."""


class LLMTimeoutError(LLMError):
    """Истёк таймаут обращения к модели."""


class LLMUnavailableError(LLMError):
    """Сервис недоступен или вернул ошибку."""


class LLMEmptyResponseError(LLMError):
    """Ответ модели пуст или без ожидаемого текста."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession | None = None) -> None:
        self._settings = settings
        self._session = session
        self._owns_session = session is None

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._settings.llm_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    async def complete(
        self,
        *,
        system: str,
        history: list[dict[str, str]],
        user_text: str,
        temperature: float,
        request_id: str | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        result = await self.chat(messages, temperature=temperature, request_id=request_id)
        if not result.content:
            raise LLMEmptyResponseError("empty")
        return result.content

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> ChatResult:
        """Один запрос к модели; при tools — нативный tool calling, не больше одного вызова."""
        request_id = request_id or uuid.uuid4().hex[:12]
        url = f"{self._settings.openrouter_base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "LLM start request_id=%s model=%s messages=%s tools=%s",
            request_id,
            self._settings.openrouter_model,
            len(messages),
            len(tools or []),
        )
        started = time.perf_counter()
        session = await self._get_session()
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                body = await response.json(content_type=None)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                if response.status >= 400:
                    logger.info(
                        "LLM error request_id=%s status=%s elapsed_ms=%s",
                        request_id,
                        response.status,
                        elapsed_ms,
                    )
                    raise LLMUnavailableError(f"HTTP {response.status}")
        except TimeoutError as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("LLM timeout request_id=%s elapsed_ms=%s", request_id, elapsed_ms)
            raise LLMTimeoutError("timeout") from exc
        except (aiohttp.ClientError, ValueError) as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("LLM unavailable request_id=%s elapsed_ms=%s", request_id, elapsed_ms)
            raise LLMUnavailableError("client error") from exc

        result = _extract_result(body)
        if not result.content and not result.tool_calls:
            logger.info("LLM empty request_id=%s elapsed_ms=%s", request_id, elapsed_ms)
            raise LLMEmptyResponseError("empty")
        logger.info(
            "LLM ok request_id=%s elapsed_ms=%s tool_calls=%s",
            request_id,
            elapsed_ms,
            len(result.tool_calls),
        )
        return result


def _extract_text(body: Any) -> str:
    return _extract_result(body).content


def _extract_result(body: Any) -> ChatResult:
    if not isinstance(body, dict):
        return ChatResult(content="")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ChatResult(content="")
    first = choices[0]
    if not isinstance(first, dict):
        return ChatResult(content="")
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        return ChatResult(content=text, tool_calls=_extract_tool_calls(message.get("tool_calls")))
    text = first.get("text")
    if isinstance(text, str):
        return ChatResult(content=text.strip())
    return ChatResult(content="")


def _extract_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=function["name"],
                arguments=arguments if isinstance(arguments, str) else "",
            )
        )
    return calls
