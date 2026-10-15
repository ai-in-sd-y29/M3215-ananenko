"""Единственный источник схем инструментов и ресурса MCP-сервера."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

HHMM_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WeatherInput(StrictModel):
    city: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
        Field(
            description=(
                "Название населённого пункта так, как его назвал пользователь, "
                "в именительном падеже. Если пользователь указал страну или регион, "
                "добавь их через запятую: «Тбилиси, Грузия». 1–100 символов."
            )
        ),
    ]


class WeatherLocation(StrictModel):
    name: str
    country: str | None
    admin1: str | None = Field(description="Административная область")
    latitude: float
    longitude: float


class WeatherCurrent(StrictModel):
    temperature_c: float = Field(description="Температура воздуха, °C")
    weather_code: int = Field(description="Код погоды WMO")
    description: str
    wind_speed: float
    wind_speed_unit: str
    observed_at: str = Field(description="Время наблюдения ISO 8601 со смещением")
    timezone: str = Field(description="Часовой пояс данных (IANA)")


class WeatherOption(StrictModel):
    name: str
    country: str | None
    admin1: str | None


class WeatherOutput(StrictModel):
    status: Literal["ok", "ambiguous", "not_found"]
    query: str
    source: str = "Open-Meteo"
    location: WeatherLocation | None = None
    current: WeatherCurrent | None = None
    options: list[WeatherOption] = Field(
        default_factory=list,
        description="Варианты одноимённых населённых пунктов при status=ambiguous",
    )


class ScheduleInput(StrictModel):
    date: Annotated[
        str,
        Field(
            pattern=DATE_PATTERN,
            description=(
                "Дата занятий в формате YYYY-MM-DD. Относительные даты "
                "(«сегодня», «завтра», «в воскресенье») вычисляй от текущей даты."
            ),
        ),
    ]


class Lesson(StrictModel):
    start: Annotated[str, Field(pattern=HHMM_PATTERN, description="Начало, HH:MM")]
    end: Annotated[str, Field(pattern=HHMM_PATTERN, description="Окончание, HH:MM")]
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    location: Annotated[str, StringConstraints(min_length=1, max_length=100)]


class ScheduleOutput(StrictModel):
    date: str
    weekday: str
    timezone: str
    lessons: list[Lesson] = Field(description="Пустой список — занятий нет")


class WeekDay(StrictModel):
    date: str
    weekday: str
    lessons: list[Lesson]


class WeekOutput(StrictModel):
    timezone: str
    week_start: str
    week_end: str
    days: list[WeekDay]


class ReminderInput(StrictModel):
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
        Field(
            description=(
                "О чём напомнить: суть действия без слов «напомни», даты и времени, "
                "например «отправить отчёт». 1–500 символов."
            )
        ),
    ]
    remind_at: Annotated[
        str,
        Field(
            description=(
                "Момент напоминания в будущем, ISO 8601 с явным смещением часового "
                "пояса пользователя, например 2026-10-13T18:30:00+03:00."
            ),
            json_schema_extra={"format": "date-time"},
        ),
    ]


class ReminderOutput(StrictModel):
    reminder_id: int
    text: str
    remind_at: str = Field(description="Нормализованное время ISO 8601 в зоне владельца")
    timezone: str = Field(description="IANA-зона владельца")
    created: bool = Field(description="false — повтор уже выполненного действия")


class ToolErrorDetail(StrictModel):
    code: str
    message: str


class ToolErrorPayload(StrictModel):
    error: ToolErrorDetail


def input_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema(mode="validation")


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema(mode="serialization")
