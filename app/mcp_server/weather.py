"""Текущая погода: Open-Meteo Geocoding + Forecast (без ключа API)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import aiohttp

from app.mcp_server.errors import ToolError
from app.mcp_server.schemas import (
    WeatherCurrent,
    WeatherLocation,
    WeatherOption,
    WeatherOutput,
)

logger = logging.getLogger("app.mcp_server.weather")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Одноимённый пункт считается однозначным, только если он крупнее остальных в 100 раз.
DOMINANCE_FACTOR = 100
MAX_OPTIONS = 5

WMO_DESCRIPTIONS = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь, туман",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


def weather_timeout() -> ToolError:
    return ToolError("weather_timeout", "Погодный сервис не ответил вовремя.")


def weather_unavailable() -> ToolError:
    return ToolError("weather_unavailable", "Погодный сервис временно недоступен.")


def weather_bad_response() -> ToolError:
    return ToolError("weather_bad_response", "Погодный сервис вернул некорректный ответ.")


class JsonFetcher(Protocol):
    async def get_json(self, url: str, params: dict[str, Any]) -> Any: ...


class _Retryable(Exception):
    def __init__(self, error: ToolError) -> None:
        self.error = error


class HttpJsonFetcher:
    """GET-запросы к фиксированным URL с конечным таймаутом и повтором только для чтения."""

    def __init__(self, *, timeout_seconds: float, retries: int = 1) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._retries = retries
        self._session: aiohttp.ClientSession | None = None

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def get_json(self, url: str, params: dict[str, Any]) -> Any:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        last_error = weather_unavailable()
        for attempt in range(self._retries + 1):
            try:
                async with self._session.get(url, params=params) as response:
                    if response.status >= 500:
                        raise _Retryable(weather_unavailable())
                    if response.status >= 400:
                        raise weather_unavailable()
                    try:
                        return await response.json(content_type=None)
                    except ValueError:
                        raise weather_bad_response() from None
            except _Retryable as exc:
                last_error = exc.error
            except TimeoutError:
                last_error = weather_timeout()
            except aiohttp.ClientError:
                last_error = weather_unavailable()
            logger.info("weather request failed code=%s attempt=%s", last_error.code, attempt + 1)
            if attempt < self._retries:
                await asyncio.sleep(0.3)
        raise last_error


def split_query(raw: str) -> tuple[str, str | None]:
    name, _, qualifier = raw.partition(",")
    name = " ".join(name.split())
    qualifier = " ".join(qualifier.split())
    return name, qualifier or None


def _norm(value: str | None) -> str:
    return (value or "").casefold().replace("ё", "е").strip()


def _matches_qualifier(place: dict[str, Any], qualifier: str) -> bool:
    wanted = _norm(qualifier)
    fields = (place.get("country"), place.get("admin1"), place.get("country_code"))
    return any(_norm(field) and _norm(field).startswith(wanted) for field in fields)


def _parse_places(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        raise weather_bad_response()
    results = body.get("results", [])
    if not isinstance(results, list):
        raise weather_bad_response()
    places = []
    for item in results:
        if not isinstance(item, dict):
            raise weather_bad_response()
        name = item.get("name")
        lat = item.get("latitude")
        lon = item.get("longitude")
        if not isinstance(name, str) or not _is_number(lat) or not _is_number(lon):
            raise weather_bad_response()
        if not str(item.get("feature_code", "PPL")).startswith("PPL"):
            continue
        places.append(item)
    return places


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _population(place: dict[str, Any]) -> int:
    value = place.get("population")
    return value if isinstance(value, int) and value > 0 else 0


def choose_place(
    places: list[dict[str, Any]], name: str, qualifier: str | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Возвращает (выбранное место, варианты). Первый результат выдачи не выбирается молча."""
    if qualifier:
        places = [place for place in places if _matches_qualifier(place, qualifier)]
    if not places:
        return None, []
    exact = [place for place in places if _norm(place["name"]) == _norm(name)]
    candidates = exact or places
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for place in sorted(candidates, key=_population, reverse=True):
        key = (_norm(place["name"]), _norm(place.get("country")), _norm(place.get("admin1")))
        unique.setdefault(key, place)
    ordered = list(unique.values())
    if len(ordered) == 1:
        return ordered[0], []
    top, rest = ordered[0], ordered[1:]
    top_population = _population(top)
    if top_population and all(
        _population(other) * DOMINANCE_FACTOR <= top_population for other in rest
    ):
        return top, []
    return None, ordered[:MAX_OPTIONS]


def _option(place: dict[str, Any]) -> WeatherOption:
    return WeatherOption(
        name=place["name"], country=place.get("country"), admin1=place.get("admin1")
    )


def _parse_current(body: Any) -> WeatherCurrent:
    try:
        current = body["current"]
        units = body.get("current_units") or {}
        offset = timedelta(seconds=int(body["utc_offset_seconds"]))
        tz_name = body["timezone"]
        observed = datetime.fromisoformat(current["time"]).replace(tzinfo=timezone(offset))
        temperature = current["temperature_2m"]
        code = current["weather_code"]
        wind = current["wind_speed_10m"]
        if not (_is_number(temperature) and _is_number(wind) and isinstance(code, int)):
            raise TypeError
        if not isinstance(tz_name, str):
            raise TypeError
    except (KeyError, TypeError, ValueError):
        raise weather_bad_response() from None
    return WeatherCurrent(
        temperature_c=float(temperature),
        weather_code=code,
        description=WMO_DESCRIPTIONS.get(code, f"код WMO {code}"),
        wind_speed=float(wind),
        wind_speed_unit=str(units.get("wind_speed_10m", "m/s")),
        observed_at=observed.isoformat(),
        timezone=tz_name,
    )


async def get_weather(city: str, fetcher: JsonFetcher) -> WeatherOutput:
    name, qualifier = split_query(city)
    if not name:
        raise ToolError("invalid_arguments", "Не указано название города.")
    body = await fetcher.get_json(
        GEOCODING_URL, {"name": name, "count": 10, "language": "ru", "format": "json"}
    )
    place, options = choose_place(_parse_places(body), name, qualifier)
    if place is None and not options:
        return WeatherOutput(status="not_found", query=city)
    if place is None:
        return WeatherOutput(
            status="ambiguous", query=city, options=[_option(item) for item in options]
        )
    forecast = await fetcher.get_json(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "auto",
        },
    )
    return WeatherOutput(
        status="ok",
        query=city,
        location=WeatherLocation(
            name=place["name"],
            country=place.get("country"),
            admin1=place.get("admin1"),
            latitude=float(place["latitude"]),
            longitude=float(place["longitude"]),
        ),
        current=_parse_current(forecast),
    )
