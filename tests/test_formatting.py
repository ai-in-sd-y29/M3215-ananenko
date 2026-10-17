from datetime import date

from app.services.formatting import (
    format_human_date,
    format_human_date_range,
    format_week_html,
    start_message_html,
)


def test_human_date_omits_month_and_year_when_same():
    today = date(2026, 10, 17)
    assert format_human_date(date(2026, 10, 18), today=today) == "18-го"
    assert format_human_date(date(2026, 11, 3), today=today) == "3 ноября"
    assert format_human_date(date(2027, 1, 5), today=today) == "5 января 2027"


def test_human_date_range_same_month():
    today = date(2026, 10, 17)
    assert format_human_date_range(date(2026, 10, 12), date(2026, 10, 18), today=today) == (
        "12–18-го"
    )


def test_week_html_quotes_lessons_and_keeps_empty_plain():
    data = {
        "timezone": "Europe/Moscow",
        "week_start": "2026-10-12",
        "week_end": "2026-10-18",
        "days": [
            {
                "date": "2026-10-12",
                "weekday": "понедельник",
                "lessons": [
                    {
                        "start": "13:30",
                        "end": "15:00",
                        "title": "Теория вероятностей",
                        "location": "ауд. 1405, Кронверкский",
                    },
                    {
                        "start": "15:30",
                        "end": "17:00",
                        "title": "Базы данных",
                        "location": "ауд. 2324, Кронверкский",
                    },
                ],
            },
            {"date": "2026-10-13", "weekday": "вторник", "lessons": []},
        ],
    }
    text = format_week_html(data, today=date(2026, 10, 17))
    assert "<b>Понедельник, 12-го:</b>" in text
    assert "<blockquote>" in text
    assert "<i>13:30–15:00</i>" in text
    assert "<b>Теория вероятностей</b>, <i>ауд. 1405, Кронверкский</i>" in text
    assert text.count("<blockquote>") == 1
    assert "<b>Вторник, 13-го:</b>" in text
    assert "занятий нет" in text
    # пустой день без quote
    tuesday_block = text.split("<b>Вторник, 13-го:</b>")[1]
    assert "<blockquote>" not in tuesday_block.split("<b>")[0]


def test_start_message_uses_html_emphasis():
    text = start_message_html()
    assert "<b>/study</b>" in text
    assert "<b>/week</b>" in text
    assert "<i>расписание на текущую неделю</i>" in text
