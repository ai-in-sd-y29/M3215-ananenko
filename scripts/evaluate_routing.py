"""Оценка маршрутизации Lab 2 на docs/evaluation/lab2-routing.json.

python -m scripts.evaluate_routing [--label baseline] [--cases weather_2,clarify_1]

Схемы инструментов берутся из настоящего MCP-сервера (stdio), модель — из .env.
Оценивается первое решение агента и нормализованные аргументы; напоминания не создаются.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from app.agent.loop import Agent, AgentContext, CallRejectedError
from app.agent.mcp_client import MCPToolbox, server_parameters, stdio_connector
from app.config import PROJECT_ROOT, Settings
from app.llm.client import LLMClient

DATASET = PROJECT_ROOT / "docs" / "evaluation" / "lab2-routing.json"
RESULTS_DIR = PROJECT_ROOT / "docs" / "evaluation" / "results"
TOOL_ACTIONS = {"get_weather", "get_schedule", "add_reminder"}


def _norm_text(value: Any) -> str:
    text = str(value).casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s,]", " ", text)
    return " ".join(text.split()).strip(" ,")


def arguments_match(action: str, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if action == "get_weather":
        return _norm_text(expected["city"]) == _norm_text(actual.get("city", ""))
    if action == "get_schedule":
        return expected["date"] == actual.get("date")
    if action == "add_reminder":
        try:
            want = datetime.fromisoformat(expected["remind_at"])
            got = datetime.fromisoformat(actual.get("remind_at", ""))
        except ValueError:
            return False
        same_time = want == got and want.utcoffset() == got.utcoffset()
        return same_time and _norm_text(expected["text"]) == _norm_text(actual.get("text", ""))
    return True


async def evaluate(settings: Settings, cases: list[dict[str, Any]], now: datetime, tz: str):
    toolbox = MCPToolbox(stdio_connector(server_parameters(settings, "")))
    llm = LLMClient(settings)
    await toolbox.start()
    if not toolbox.available:
        raise SystemExit("MCP-сервер не запустился — оценка невозможна.")
    agent = Agent(llm, toolbox, None, schedule_timezone=tz, clock=lambda: now)
    rows = []
    try:
        for case in cases:
            ctx = AgentContext(
                chat_id=0,
                source_key=f"eval:{case['id']}",
                user_text=case["query"],
                mode="study",
                temperature=0.0,
                timezone=tz,
            )
            decision = await agent.decide(
                agent.initial_messages(ctx, now),
                agent.tool_definitions(),
                temperature=0.0,
                request_id=f"eval-{case['id']}",
            )
            actual_action = decision.call.name if decision.kind == "call_tool" else decision.kind
            raw_args: dict[str, Any] = {}
            normalized: dict[str, Any] = {}
            note = ""
            if decision.kind == "call_tool":
                try:
                    raw_args = json.loads(decision.call.arguments or "{}")
                    allowed = {spec.name for spec in toolbox.tools}
                    normalized = agent.validate_call(decision.call, ctx, now, allowed)
                except CallRejectedError as exc:
                    note = f"проверка: {exc.code}"
                except ValueError:
                    note = "аргументы не JSON"
            elif decision.kind == "clarify":
                note = decision.text
            expected_action = case["expected_action"]
            action_ok = actual_action == expected_action
            args_ok = None
            if expected_action in TOOL_ACTIONS:
                args_ok = action_ok and arguments_match(
                    expected_action, case["expected_arguments"], normalized
                )
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_action": expected_action,
                    "actual_action": actual_action,
                    "expected_arguments": case["expected_arguments"],
                    "model_arguments": raw_args,
                    "normalized_arguments": normalized,
                    "action_ok": action_ok,
                    "arguments_ok": args_ok,
                    "note": note,
                }
            )
            print(f"{case['id']}: {actual_action} {normalized or ''} {note}".rstrip())
    finally:
        await toolbox.stop()
        await llm.aclose()
    return rows


def to_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        f"# Оценка маршрутизации: {meta['label']}",
        "",
        f"Модель: `{meta['model']}`, temperature: {meta['temperature']}, "
        f"время эксперимента: {meta['now']} ({meta['timezone']}), запуск: {meta['run_at']}.",
        "",
        "| № | Запрос | Ожидаемое действие | Фактическое действие | Аргументы | "
        "Аргументы корректны | Результат |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(rows, 1):
        args = json.dumps(row["normalized_arguments"], ensure_ascii=False) or ""
        if row["note"]:
            args = f"{args if row['normalized_arguments'] else ''} {row['note']}".strip()
        args_ok = {True: "да", False: "нет", None: "—"}[row["arguments_ok"]]
        verdict = "✅" if row["action_ok"] and row["arguments_ok"] is not False else "❌"
        lines.append(
            f"| {index} | {row['query']} | {row['expected_action']} | {row['actual_action']} "
            f"| {args.replace('|', '/')} | {args_ok} | {verdict} |"
        )
    lines += [
        "",
        f"routing_accuracy = {meta['routing_correct']}/{meta['routing_total']} "
        f"= {meta['routing_accuracy']:.2f}",
        f"argument_accuracy = {meta['args_correct']}/{meta['args_total']} "
        f"= {meta['argument_accuracy']:.2f}",
    ]
    return "\n".join(lines) + "\n"


def evaluation_settings(env_file: str) -> Settings:
    """Для оценки нужен только OpenRouter: токен бота и БД не используются."""
    values = {**dotenv_values(env_file, interpolate=False), **os.environ}
    key = values.get("OPENROUTER_API_KEY") or ""
    if not key:
        raise SystemExit("Задайте OPENROUTER_API_KEY в окружении или в .env.")
    defaults = Settings(bot_token="unused", postgres_password="unused")
    return replace(
        defaults,
        openrouter_api_key=key,
        openrouter_base_url=(
            values.get("OPENROUTER_BASE_URL") or defaults.openrouter_base_url
        ).rstrip("/"),
        openrouter_model=values.get("OPENROUTER_MODEL") or defaults.openrouter_model,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--cases", default="", help="id кейсов через запятую (по умолчанию все)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = evaluation_settings(args.env_file)
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(dataset["evaluation_context"]["now"])
    tz = dataset["evaluation_context"]["timezone"]
    wanted = {item for item in args.cases.split(",") if item}
    cases = [case for case in dataset["cases"] if not wanted or case["id"] in wanted]

    rows = asyncio.run(evaluate(settings, cases, now, tz))
    tool_rows = [row for row in rows if row["expected_action"] in TOOL_ACTIONS]
    meta = {
        "label": args.label,
        "model": settings.openrouter_model,
        "temperature": 0.0,
        "now": now.isoformat(),
        "timezone": tz,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "routing_correct": sum(row["action_ok"] for row in rows),
        "routing_total": len(rows),
        "args_correct": sum(bool(row["arguments_ok"]) for row in tool_rows),
        "args_total": len(tool_rows),
    }
    meta["routing_accuracy"] = meta["routing_correct"] / max(1, meta["routing_total"])
    meta["argument_accuracy"] = meta["args_correct"] / max(1, meta["args_total"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = RESULTS_DIR / f"lab2-routing-{args.label}"
    Path(f"{base}.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = to_markdown(rows, meta)
    Path(f"{base}.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
