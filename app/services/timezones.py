"""Локальное время напоминаний: интерпретация только в сохранённой IANA-зоне."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


class LocalTimeError(ValueError):
    code = "invalid_datetime"


class InvalidDateTimeError(LocalTimeError):
    code = "invalid_datetime"


class AmbiguousLocalTimeError(LocalTimeError):
    code = "ambiguous_local_time"


class NonexistentLocalTimeError(LocalTimeError):
    code = "nonexistent_local_time"


def localize(wall: datetime, tz: ZoneInfo) -> datetime:
    """Локальное время без смещения → aware; переход на летнее время требует уточнения."""
    wall = wall.replace(tzinfo=None, fold=0)
    first = wall.replace(tzinfo=tz, fold=0)
    second = wall.replace(tzinfo=tz, fold=1)
    exists = any(
        candidate.astimezone(UTC).astimezone(tz).replace(tzinfo=None) == wall
        for candidate in (first, second)
    )
    if not exists:
        raise NonexistentLocalTimeError(wall.isoformat())
    if first.utcoffset() != second.utcoffset():
        raise AmbiguousLocalTimeError(wall.isoformat())
    return first


def normalize_remind_at(raw: str, tz_name: str) -> datetime:
    """Время напоминания из решения модели → aware datetime в зоне пользователя.

    Если смещение совпадает с одним из смещений зоны для этого локального времени,
    авторитетно локальное время. Иное смещение (например, Z) трактуется как момент
    времени и переводится в зону пользователя.
    """
    try:
        moment = datetime.fromisoformat(raw.strip())
    except (ValueError, AttributeError):
        raise InvalidDateTimeError(str(raw)[:40]) from None
    tz = ZoneInfo(tz_name)
    if moment.tzinfo is None or moment.utcoffset() is None:
        return localize(moment, tz)
    wall = moment.replace(tzinfo=None)
    zone_offsets = {wall.replace(tzinfo=tz, fold=fold).utcoffset() for fold in (0, 1)}
    if moment.utcoffset() in zone_offsets:
        return localize(wall, tz)
    converted = moment.astimezone(tz)
    localize(converted.replace(tzinfo=None), tz)
    return converted


def offset_label(moment: datetime) -> str:
    offset = moment.utcoffset()
    if offset is None:
        return "UTC"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    hours, minutes = divmod(abs(total) // 60, 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"
