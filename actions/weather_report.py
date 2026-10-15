"""
Real weather data from Open-Meteo (free, no API key).

City comes from the request, else the user's saved city in memory, else the
approximate location of this machine's IP address. Shared helpers here are also
used by actions/world_monitor.py for severe-weather alerts.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from urllib.parse import quote_plus

_UA = {"User-Agent": "JARVIS-MarkLV/1.0"}

# WMO weather interpretation codes → words
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _get_json(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def default_city() -> str:
    """User's saved city (memory) → IP geolocation → ''."""
    try:
        from memory.memory_manager import load_memory
        ident = load_memory().get("identity", {})
        for key in ("city", "location", "home_city", "hometown"):
            v = ident.get(key)
            v = v.get("value") if isinstance(v, dict) else v
            if v:
                return str(v)
    except Exception:
        pass
    try:
        return _get_json("https://ipapi.co/json/", 5).get("city", "") or ""
    except Exception:
        return ""


def geocode(city: str) -> dict | None:
    url = ("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&name="
           + urllib.parse.quote(city.split(",")[0].strip()))
    res = (_get_json(url).get("results") or [None])[0]
    if not res:
        return None
    return {"name": res["name"], "country": res.get("country", ""),
            "lat": res["latitude"], "lon": res["longitude"]}


def forecast(lat: float, lon: float, days: int = 3) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&timezone=auto&forecast_days={days}"
        "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
        "weather_code,wind_speed_10m,precipitation"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max,precipitation_sum,wind_speed_10m_max,uv_index_max"
    )
    return _get_json(url)


def severe_flags(daily: dict, i: int) -> list[str]:
    """Human-readable warnings for day i of an Open-Meteo daily block."""
    flags = []
    code = daily["weather_code"][i]
    tmax, tmin = daily["temperature_2m_max"][i], daily["temperature_2m_min"][i]
    rain_p = daily.get("precipitation_probability_max", [None] * 9)[i] or 0
    rain_mm = daily.get("precipitation_sum", [0] * 9)[i] or 0
    wind = daily.get("wind_speed_10m_max", [0] * 9)[i] or 0
    uv = daily.get("uv_index_max", [0] * 9)[i] or 0
    if code >= 95:
        flags.append(_WMO.get(code, "thunderstorm"))
    if rain_mm >= 20 or (code in (65, 67, 82) and rain_p >= 60):
        flags.append(f"heavy rain ({rain_mm:.0f} mm, {rain_p:.0f}% chance)")
    if code in (75, 86):
        flags.append("heavy snow")
    if tmax >= 40:
        flags.append(f"extreme heat ({tmax:.0f}°C)")
    if tmin <= 2:
        flags.append(f"near-freezing cold ({tmin:.0f}°C)")
    if wind >= 50:
        flags.append(f"strong wind ({wind:.0f} km/h)")
    if uv >= 10:
        flags.append(f"extreme UV (index {uv:.0f})")
    return flags


def report(city: str = "", when: str = "today") -> str:
    city = (city or "").strip() or default_city()
    if not city:
        return "I don't know which city to check — tell me the city name."
    place = geocode(city)
    if not place:
        return f"I couldn't find a place called '{city}'."
    data = forecast(place["lat"], place["lon"], 3)
    cur, daily = data["current"], data["daily"]
    w = (when or "today").lower()
    idx = 1 if "tomorrow" in w or "kal" in w else 2 if "day after" in w or "parso" in w else 0

    lines = [f"Weather for {place['name']}, {place['country']}:"]
    if idx == 0:
        lines.append(
            f"Now: {cur['temperature_2m']:.0f}°C (feels {cur['apparent_temperature']:.0f}°C), "
            f"{_WMO.get(cur['weather_code'], 'unknown')}, humidity {cur['relative_humidity_2m']}%, "
            f"wind {cur['wind_speed_10m']:.0f} km/h."
        )
    for i in range(idx, min(idx + (3 if "week" in w or "next days" in w else 1), len(daily["time"]))):
        day = datetime.strptime(daily["time"][i], "%Y-%m-%d").strftime("%A %d %b")
        lines.append(
            f"{day}: {_WMO.get(daily['weather_code'][i], 'unknown')}, "
            f"{daily['temperature_2m_min'][i]:.0f}–{daily['temperature_2m_max'][i]:.0f}°C, "
            f"rain chance {daily['precipitation_probability_max'][i] or 0:.0f}% "
            f"({daily['precipitation_sum'][i] or 0:.1f} mm), "
            f"max wind {daily['wind_speed_10m_max'][i]:.0f} km/h, UV {daily['uv_index_max'][i] or 0:.0f}."
        )
        flags = severe_flags(daily, i)
        if flags:
            lines.append("  ⚠ Warning: " + ", ".join(flags))
    return "\n".join(lines)


def weather_action(
    parameters: dict,
    player=None,
    session_memory=None,
) -> str:
    p    = parameters or {}
    city = str(p.get("city") or "").strip()
    when = str(p.get("time") or p.get("when") or "today").strip()
    try:
        msg = report(city, when)
    except Exception as e:
        # Network trouble: fall back to showing it in the browser.
        q = f"weather in {city or default_city()} {when}"
        try:
            webbrowser.open(f"https://www.google.com/search?q={quote_plus(q)}")
        except Exception:
            pass
        msg = f"Live weather data unavailable ({e}); opened a weather search instead."
    _log(msg, player)
    if player:
        try:
            player.show_content("WEATHER", msg)
        except Exception:
            pass
    return msg


def _log(message: str, player=None) -> None:
    print(f"[Weather] {message.splitlines()[0]}")


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "weather_report",
    "description": (
        "Real weather data: current conditions and forecast (temperature, rain chance, wind, UV) "
        "with severe-weather warnings, for today, tomorrow, the day after or the next 3 days. "
        "City defaults to the user's own city when not given."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "city": {"type": "STRING", "description": "City name (optional — defaults to the user's city)"},
            "time": {"type": "STRING", "description": "today (default) | tomorrow | day after tomorrow | week"},
        },
        "required": [],
    },
    "handler": weather_action,
}
