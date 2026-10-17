"""MCP-клиент: подключение к собственному серверу, discovery, вызовы с таймаутом.

Соединением владеет одна фоновая задача: anyio-контексты stdio_client должны
открываться и закрываться в одной задаче. Обработчики только отправляют запросы.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import anyio
from mcp import ClientSession, McpError, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from app.agent.errors import MCPBadResultError, MCPError, MCPToolError, MCPUnavailableError
from app.config import PROJECT_ROOT, Settings

logger = logging.getLogger("app.agent.mcp")

Connector = Callable[[], AbstractAsyncContextManager[ClientSession]]
REQUEST_TIMEOUT_CODE = 408


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    read_only: bool

    @property
    def needs_confirmation(self) -> bool:
        return not self.read_only

    def as_llm_function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ResourceSpec:
    uri: str
    name: str
    description: str


def tool_spec_from_mcp(tool: types.Tool) -> ToolSpec:
    annotations = tool.annotations
    # Без явного readOnlyHint=True инструмент считается изменяющим (нужно подтверждение).
    read_only = bool(annotations is not None and annotations.readOnlyHint is True)
    if tool.name == "add_reminder":
        read_only = False
    return ToolSpec(
        name=tool.name,
        description=tool.description or tool.title or tool.name,
        input_schema=tool.inputSchema,
        output_schema=tool.outputSchema,
        read_only=read_only,
    )


def server_parameters(settings: Settings, owner_secret: str) -> StdioServerParameters:
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "SCHEDULE_PATH": settings.schedule_path,
        "SCHEDULE_TIMEZONE": settings.schedule_timezone,
        "WEATHER_TIMEOUT_SECONDS": str(settings.weather_timeout_seconds),
        "POSTGRES_HOST": settings.postgres_host,
        "POSTGRES_PORT": str(settings.postgres_port),
        "POSTGRES_DB": settings.postgres_db,
        "POSTGRES_USER": settings.postgres_user,
        "POSTGRES_PASSWORD": settings.postgres_password,
        "MCP_OWNER_SECRET": owner_secret,
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


def stdio_connector(params: StdioServerParameters) -> Connector:
    @asynccontextmanager
    async def connect() -> AsyncIterator[ClientSession]:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    return connect


def _describe(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(inner) for inner in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"[:300]


class MCPToolbox:
    def __init__(
        self,
        connector: Connector,
        *,
        call_timeout_seconds: float = 25.0,
        connect_timeout_seconds: float = 20.0,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        self._connector = connector
        self._call_timeout = call_timeout_seconds
        self._connect_timeout = connect_timeout_seconds
        self._max_backoff = max_backoff_seconds
        self._session: ClientSession | None = None
        self._tools: dict[str, ToolSpec] = {}
        self._resources: list[ResourceSpec] = []
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._attempted = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._stopping = False

    @property
    def available(self) -> bool:
        return self._session is not None

    @property
    def tools(self) -> list[ToolSpec]:
        return list(self._tools.values()) if self.available else []

    @property
    def resources(self) -> list[ResourceSpec]:
        return list(self._resources) if self.available else []

    def spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name) if self.available else None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="mcp-client")
        try:
            async with asyncio.timeout(self._connect_timeout):
                await self._attempted.wait()
        except TimeoutError:
            logger.warning("MCP-сервер не ответил за %s с при запуске", self._connect_timeout)

    async def stop(self) -> None:
        self._stopping = True
        self._stop_event.set()
        self._wake.set()
        if self._task is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(self._task, return_exceptions=True)
            except TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    def _mark_broken(self) -> None:
        self._wake.set()

    async def _run(self) -> None:
        delay = 1.0
        while not self._stopping:
            self._wake.clear()
            try:
                async with self._connector() as session:
                    listed = await session.list_tools()
                    resources = await session.list_resources()
                    self._tools = {tool.name: tool_spec_from_mcp(tool) for tool in listed.tools}
                    self._resources = [
                        ResourceSpec(
                            uri=str(item.uri),
                            name=item.title or item.name,
                            description=item.description or "",
                        )
                        for item in resources.resources
                    ]
                    self._session = session
                    self._attempted.set()
                    delay = 1.0
                    logger.info(
                        "MCP подключён: tools=%s resources=%s",
                        ",".join(self._tools),
                        ",".join(item.uri for item in self._resources),
                    )
                    await self._wake.wait()
            except Exception as exc:
                logger.warning("MCP-сервер недоступен: %s", _describe(exc))
            finally:
                self._session = None
                self._attempted.set()
            if self._stopping:
                break
            try:
                async with asyncio.timeout(delay):
                    await self._stop_event.wait()
            except TimeoutError:
                pass
            delay = min(delay * 2, self._max_backoff)

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        session = self._session
        if session is None:
            raise MCPUnavailableError("mcp_unavailable")
        if name not in self._tools:
            raise MCPToolError("unknown_tool", "tool is not discovered")
        try:
            result = await session.call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(seconds=self._call_timeout),
                meta=meta,
            )
        except McpError as exc:
            self._mark_broken()
            code = "mcp_timeout" if exc.error.code == REQUEST_TIMEOUT_CODE else "mcp_unavailable"
            raise MCPUnavailableError(code, exc.error.message) from None
        except RuntimeError as exc:
            # ClientSession проверяет structuredContent по outputSchema.
            raise MCPBadResultError("bad_result", str(exc)[:200]) from None
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, OSError) as exc:
            self._mark_broken()
            raise MCPUnavailableError("mcp_unavailable", type(exc).__name__) from None
        return parse_tool_result(result)

    async def read_resource(self, uri: str) -> dict[str, Any]:
        session = self._session
        if session is None:
            raise MCPUnavailableError("mcp_unavailable")
        try:
            async with asyncio.timeout(self._call_timeout):
                result = await session.read_resource(AnyUrl(uri))
        except TimeoutError:
            self._mark_broken()
            raise MCPUnavailableError("mcp_timeout") from None
        except McpError as exc:
            raise MCPToolError("resource_error", exc.error.message) from None
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, OSError) as exc:
            self._mark_broken()
            raise MCPUnavailableError("mcp_unavailable", type(exc).__name__) from None
        for content in result.contents:
            if isinstance(content, types.TextResourceContents):
                try:
                    data = json.loads(content.text)
                except ValueError:
                    break
                if isinstance(data, dict):
                    return data
        raise MCPBadResultError("bad_result", "resource is not a JSON object")


def parse_tool_result(result: types.CallToolResult) -> dict[str, Any]:
    if result.isError:
        error = (result.structuredContent or {}).get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            raise MCPToolError(error["code"], str(error.get("message", "")))
        text = " ".join(
            block.text for block in result.content if isinstance(block, types.TextContent)
        )
        code = "invalid_arguments" if "validation error" in text.lower() else "tool_error"
        raise MCPToolError(code, text[:200])
    if not isinstance(result.structuredContent, dict):
        raise MCPBadResultError("bad_result", "structuredContent is missing")
    return result.structuredContent


__all__ = [
    "MCPError",
    "MCPToolbox",
    "ResourceSpec",
    "ToolSpec",
    "parse_tool_result",
    "server_parameters",
    "stdio_connector",
]
