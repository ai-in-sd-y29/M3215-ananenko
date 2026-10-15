"""Расписание из файла конфигурации: двухнедельный цикл в пределах семестра."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from app.mcp_server.errors import ToolError
from app.mcp_server.schemas import Lesson, ScheduleOutput, WeekDay, WeekOutput

WEEKDAY_KEYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
WEEKDAY_NAMES = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)


class _CycleWeek(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    days: dict[str, list[Lesson]]


class _ScheduleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    valid_from: date
    valid_to: date
    cycle_start: date
    weeks: list[_CycleWeek]


@dataclass(frozen=True)
class Schedule:
    valid_from: date
    valid_to: date
    cycle_start: date
    weeks: tuple[dict[str, tuple[Lesson, ...]], ...]

    def lessons_on(self, day: date) -> list[Lesson]:
        if not self.valid_from <= day <= self.valid_to or not self.weeks:
            return []
        week_index = ((day - self.cycle_start).days // 7) % len(self.weeks)
        lessons = self.weeks[week_index].get(WEEKDAY_KEYS[day.weekday()], ())
        return sorted(lessons, key=lambda lesson: lesson.start)


def schedule_unavailable() -> ToolError:
    return ToolError("schedule_unavailable", "Источник расписания недоступен.")


def parse_schedule(raw: Any) -> Schedule:
    try:
        data = _ScheduleFile.model_validate(raw)
    except ValidationError:
        raise schedule_unavailable() from None
    if data.cycle_start.weekday() != 0 or data.valid_from > data.valid_to:
        raise schedule_unavailable()
    weeks = []
    for week in data.weeks:
        if set(week.days) - set(WEEKDAY_KEYS):
            raise schedule_unavailable()
        weeks.append({key: tuple(value) for key, value in week.days.items()})
    return Schedule(
        valid_from=data.valid_from,
        valid_to=data.valid_to,
        cycle_start=data.cycle_start,
        weeks=tuple(weeks),
    )


def load_schedule(path: Path | str) -> Schedule:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise schedule_unavailable() from None
    return parse_schedule(raw)


def parse_day(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise ToolError(
            "invalid_arguments", "Дата должна быть существующей и в формате YYYY-MM-DD."
        ) from None


def schedule_for_day(schedule: Schedule, day: date, tz_name: str) -> ScheduleOutput:
    return ScheduleOutput(
        date=day.isoformat(),
        weekday=WEEKDAY_NAMES[day.weekday()],
        timezone=tz_name,
        lessons=schedule.lessons_on(day),
    )


def build_week(schedule: Schedule, today: date, tz_name: str) -> WeekOutput:
    """Неделя пн–вс, содержащая today; today вычисляет вызывающий код в нужной зоне."""
    monday = today - timedelta(days=today.weekday())
    days = []
    for offset in range(7):
        day = monday + timedelta(days=offset)
        days.append(
            WeekDay(
                date=day.isoformat(),
                weekday=WEEKDAY_NAMES[day.weekday()],
                lessons=schedule.lessons_on(day),
            )
        )
    return WeekOutput(
        timezone=tz_name,
        week_start=monday.isoformat(),
        week_end=(monday + timedelta(days=6)).isoformat(),
        days=days,
    )
