"""
World Monitor — news, weather, tech/AI updates and important alerts.

On demand (tool `world_monitor`):
    briefing   world headlines + tech/AI headlines + weather for the user's city
    world      top world / country news
    tech       latest AI & technology news
    alerts     run the alert check right now

In the background (main.py calls check_world() every 30 minutes):
    • severe weather for the user's city today/tomorrow (thunderstorm, heavy
      rain, extreme heat, strong wind…) — once per day per warning
    • breaking news matching serious keywords (earthquake, cyclone, flood,
      attack, outbreak…) for the world and the user's country — each headline once
    • a daily AI/tech digest (3 new headlines) after 09:00
State lives in memory/long_term.json under "world_monitor".
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

_SERIOUS = re.compile(
    r"\b(earthquake|tsunami|cyclone|hurricane|typhoon|tornado|flood(s|ing)?|landslide|"
    r"wildfire|heatwave|heat wave|red alert|orange alert|evacuat\w*|"
    r"explosion|blast|terror\w*|attack|shooting|war|missile|airstrike|"
    r"outbreak|pandemic|epidemic|lockdown|curfew|"
    r"crash(es|ed)?|derail\w*|collapse[ds]?|stampede|"
    r"outage|blackout|cyberattack|data breach|"
    r"emergency|disaster|death toll|killed)\b",
    re.IGNORECASE,
)

# World feed: only events that matter anywhere. Country feed: _SERIOUS.
_MAJOR = re.compile(
    r"\b(earthquake|tsunami|cyclone|hurricane|typhoon|volcan\w*|"
    r"war|invasion|missile strike|airstrikes?|nuclear|ceasefire|"
    r"pandemic|outbreak|global outage|terror attack)\b",
    re.IGNORECASE,
)

_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "is", "are", "with", "as",
         "at", "by", "says", "said", "after", "from", "its", "it", "be", "new", "over"}

_TECH_QUERIES = ("AI startup launches new model", "OpenAI Google Anthropic AI", "tech industry news")


def _hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode("utf-8", "ignore")).hexdigest()[:12]


def _load_state() -> dict:
    from memory.memory_manager import load_memory
    st = load_memory().get("world_monitor", {})
    return st if isinstance(st, dict) else {}


def _save_state(state: dict) -> None:
    from memory.memory_manager import load_memory, MEMORY_PATH, _lock
    memory = load_memory()
    state["seen"] = state.get("seen", [])[-300:]
    memory["world_monitor"] = state
    with _lock:
        MEMORY_PATH.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")


def _news(query: str, n: int = 8) -> list[dict]:
    from actions.web_search import _ddg_news
    try:
        return [r for r in _ddg_news(query, max_results=n) if r.get("title")]
    except Exception as e:
        print(f"[WorldMonitor] news failed for '{query}': {e}")
        return []


def _country() -> str:
    try:
        from actions.weather_report import _get_json
        return _get_json("https://ipapi.co/json/", 5).get("country_name", "") or ""
    except Exception:
        return ""


def _fmt(items: list[dict], n: int) -> str:
    lines = []
    for r in items[:n]:
        src = f" ({r['source']})" if r.get("source") else ""
        lines.append(f"• {r['title'].strip()}{src}")
    return "\n".join(lines) if lines else "• nothing found"


def _words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in _STOP and len(w) > 2}


def _dedupe(items: list[dict]) -> list[dict]:
    """Drops repeats of the same story (titles sharing most of their words)."""
    kept: list[tuple[set, dict]] = []
    for r in items:
        w = _words(r["title"])
        if any(len(w & k) / max(1, min(len(w), len(k))) >= 0.5 for k, _ in kept):
            continue
        kept.append((w, r))
    return [r for _, r in kept]


def tech_news(n: int = 5) -> str:
    items = []
    for q in _TECH_QUERIES[:2]:
        items += _news(q, 6)
    return "Tech & AI:\n" + _fmt(_dedupe(items), n)


def world_news(n: int = 5) -> str:
    country = _country()
    items = _news("international news", 8)
    text = "World:\n" + _fmt(_dedupe(items), n)
    if country:
        text += f"\n{country}:\n" + _fmt(_dedupe(_news(f"{country} news today", 6)), 3)
    return text


def briefing() -> str:
    from actions.weather_report import report
    parts = []
    try:
        parts.append(report("", "today"))
    except Exception as e:
        parts.append(f"Weather unavailable: {e}")
    parts.append(world_news(4))
    parts.append(tech_news(4))
    return "\n\n".join(parts)


# ── Background alerts ────────────────────────────────────────────────────────

def _weather_alerts(state: dict) -> list[str]:
    from actions.weather_report import default_city, geocode, forecast, severe_flags
    city = default_city()
    if not city:
        return []
    place = geocode(city)
    if not place:
        return []
    daily = forecast(place["lat"], place["lon"], 2)["daily"]
    out = []
    for i, label in ((0, "today"), (1, "tomorrow")):
        flags = severe_flags(daily, i)
        if not flags:
            continue
        key = f"{daily['time'][i]}:{','.join(sorted(flags))}"
        if key in state.get("weather_sent", []):
            continue
        state.setdefault("weather_sent", []).append(key)
        state["weather_sent"] = state["weather_sent"][-20:]
        out.append(f"[WORLD_ALERT] Weather warning for {place['name']} {label}: "
                   + ", ".join(flags) + ".")
    return out


def _breaking_alerts(state: dict) -> list[str]:
    seen = set(state.get("seen", []))
    country = _country()
    feeds = [("international news", _MAJOR)]
    if country:
        feeds.append((f"{country} breaking news", _SERIOUS))
    out = []
    for q, pattern in feeds:
        for r in _dedupe(_news(q, 10)):
            title = r["title"].strip()
            h = _hash(title[:70])
            if h in seen:
                continue
            seen.add(h)
            state.setdefault("seen", []).append(h)
            if pattern.search(title):
                src = f" (source: {r['source']})" if r.get("source") else ""
                out.append(f"[WORLD_ALERT] Breaking: {title}{src}")
    # On the very first run everything is "new" — remember it, alert nothing.
    if not state.get("primed"):
        state["primed"] = True
        return []
    return out[:2]


def _tech_digest(state: dict) -> list[str]:
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.hour < 9 or state.get("digest_date") == today:
        return []
    seen = set(state.get("tech_seen", []))
    fresh = []
    for q in _TECH_QUERIES[:2]:
        for r in _news(q, 8):
            h = _hash(r["title"][:70])
            if h not in seen:
                seen.add(h)
                fresh.append(r)
    state["tech_seen"] = list(seen)[-200:]
    state["digest_date"] = today
    if not fresh:
        return []
    return ["[WORLD_ALERT] Today's AI/tech digest:\n" + _fmt(_dedupe(fresh), 3)]


def check_world() -> list[str]:
    """Runs all background checks; returns alert strings (empty when nothing new)."""
    state = _load_state()
    alerts: list[str] = []
    for fn in (_weather_alerts, _breaking_alerts, _tech_digest):
        try:
            alerts += fn(state)
        except Exception as e:
            print(f"[WorldMonitor] {fn.__name__} failed: {e}")
    try:
        _save_state(state)
    except Exception as e:
        print(f"[WorldMonitor] could not save state: {e}")
    return alerts


def world_monitor(parameters: dict, player=None, **_) -> str:
    action = str((parameters or {}).get("action") or "briefing").lower().strip()
    if action == "tech":
        text, title = tech_news(6), "TECH & AI NEWS"
    elif action in ("world", "news"):
        text, title = world_news(6), "WORLD NEWS"
    elif action == "alerts":
        alerts = check_world()
        text, title = ("\n\n".join(a.replace("[WORLD_ALERT] ", "") for a in alerts)
                       or "No new alerts right now."), "ALERTS"
    else:
        text, title = briefing(), "WORLD BRIEFING"
    if player:
        try:
            player.show_content(title, text)
        except Exception:
            pass
    return text


TOOL = {
    "name": "world_monitor",
    "description": (
        "World Monitor: a briefing of weather + world news + AI/tech news ('what's happening', "
        "'morning briefing', 'aaj ki khabar'), the latest AI & technology updates, world/country "
        "headlines, or an immediate check for important alerts (severe weather, breaking "
        "disasters/incidents). Background alerts also run automatically."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": "briefing (default) | world | tech | alerts"},
        },
        "required": [],
    },
    "handler": world_monitor,
}
