"""Человекочитаемые даты и HTML-оформление заготовленных ответов для Telegram."""

from __future__ import annotations

import html
from datetime import date, datetime

MONTHS_GENITIVE = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

WEEKDAYS_RU = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)


def format_human_date(value: date | datetime, *, today: date | None = None) -> str:
    """Дата для пользователя: «18-го», «18 октября» или «18 октября 2027».

    Год — только если отличается от текущего; месяц — только если отличается
    от текущего месяца (при том же годе).
    """
    day = value.date() if isinstance(value, datetime) else value
    today = today or date.today()
    if day.year != today.year:
        return f"{day.day} {MONTHS_GENITIVE[day.month - 1]} {day.year}"
    if day.month != today.month:
        return f"{day.day} {MONTHS_GENITIVE[day.month - 1]}"
    return f"{day.day}-го"


def format_human_date_range(start: date, end: date, *, today: date | None = None) -> str:
    """Диапазон дат для заголовка недели."""
    today = today or date.today()
    if start == end:
        return format_human_date(start, today=today)
    if start.year == end.year == today.year and start.month == end.month == today.month:
        return f"{start.day}–{end.day}-го"
    if start.year == end.year == today.year and start.month == end.month:
        month = MONTHS_GENITIVE[start.month - 1]
        return f"{start.day}–{end.day} {month}"
    if start.year == end.year:
        left = format_human_date(start, today=date(start.year, start.month, 1))
        right = format_human_date(end, today=date(end.year, end.month, 1))
        # Оба с месяцем, без года (год общий и текущий) или с годом.
        if start.year == today.year:
            left_m = MONTHS_GENITIVE[start.month - 1]
            right_m = MONTHS_GENITIVE[end.month - 1]
            return f"{start.day} {left_m} — {end.day} {right_m}"
        return f"{left} — {right}"
    return f"{format_human_date(start, today=today)} — {format_human_date(end, today=today)}"


def format_human_datetime(moment: datetime, *, today: date | None = None) -> str:
    return f"{format_human_date(moment, today=today)} {moment:%H:%M}"


def _parse_iso_date(raw: str) -> date:
    return date.fromisoformat(raw[:10])


def format_week_html(data: dict, *, today: date | None = None) -> str:
    """Расписание недели: день и предмет жирным, время и аудитория курсивом, день в quote."""
    today = today or date.today()
    start = _parse_iso_date(data["week_start"])
    end = _parse_iso_date(data["week_end"])
    tz = html.escape(str(data.get("timezone", "")))
    span = html.escape(format_human_date_range(start, end, today=today))
    lines = [f"<b>Расписание на неделю {span}</b> <i>({tz})</i>:"]
    for day in data["days"]:
        day_date = _parse_iso_date(day["date"])
        weekday = html.escape(str(day["weekday"]).capitalize())
        human = html.escape(format_human_date(day_date, today=today))
        lines.append("")
        lines.append(f"<b>{weekday}, {human}:</b>")
        lessons = day.get("lessons") or []
        if not lessons:
            lines.append("занятий нет")
            continue
        blocks: list[str] = []
        for lesson in lessons:
            time_range = html.escape(f"{lesson['start']}–{lesson['end']}")
            title = html.escape(str(lesson["title"]))
            location = html.escape(str(lesson["location"]))
            blocks.append(f"<i>{time_range}</i>\n<b>{title}</b>, <i>{location}</i>")
        lines.append("<blockquote>" + "\n\n".join(blocks) + "</blockquote>")
    return "\n".join(lines)


def start_message_html() -> str:
    return (
        "Я <b>AI-ассистент студента</b>.\n\n"
        "<b>Команды:</b>\n"
        "<b>/study</b> — <i>объяснение учебных вопросов по программированию</i>\n"
        "<b>/translate</b> — <i>перевод текста на указанный язык</i>\n"
        "<b>/exam</b> — <i>экзаменатор: вопросы по теме и проверка ответов</i>\n"
        "<b>/settings</b> — <i>режим, модель и temperature</i>\n"
        "<b>/reset</b> — <i>очистить историю диалога</i>\n"
        "<b>/tools</b> — <i>какие действия агент может выполнить через MCP-сервер</i>\n"
        "<b>/timezone</b> <i>&lt;зона&gt;</i> — <i>часовой пояс для напоминаний, "
        "например Europe/Moscow</i>\n"
        "<b>/why</b> — <i>почему агент выбрал действие в последнем запросе</i>\n"
        "<b>/week</b> — <i>расписание на текущую неделю</i>\n\n"
        "Отправьте текстовое сообщение — отвечу в активном режиме "
        "(по умолчанию <b>/study</b>). Могу узнать погоду, найти занятия на дату и "
        "подготовить напоминание <i>(создам его только после вашего подтверждения)</i>."
    )


def tools_message_html(lines_plain: list[str]) -> str:
    """Оформляет уже собранный plain-список tools в HTML."""
    if not lines_plain:
        return "Инструменты сейчас недоступны."
    out: list[str] = []
    for line in lines_plain:
        if line.startswith("Доступные") or line.startswith("Ресурсы"):
            out.append(f"<b>{html.escape(line)}</b>")
        elif line.startswith("• "):
            body = line[2:]
            if " — " in body:
                name, rest = body.split(" — ", 1)
                out.append(f"• <b>{html.escape(name)}</b> — <i>{html.escape(rest)}</i>")
            else:
                out.append(f"• <i>{html.escape(body)}</i>")
        elif line.strip():
            out.append(html.escape(line))
        else:
            out.append("")
    return "\n".join(out)
