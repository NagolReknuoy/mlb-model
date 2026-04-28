# =============================================================================
# weather.py — Game-time weather via Open-Meteo (free, no API key)
# =============================================================================

import requests
import pandas as pd
import numpy as np
from datetime import date
from modules.utils import clean_name


# ── Stadium coordinates ───────────────────────────────────────────────────────
STADIUM_COORDS = {
    "los angeles angels":     {"lat": 33.800, "lon": -117.883, "dome": False},
    "houston astros":         {"lat": 29.757, "lon":  -95.355, "dome": True},
    "oakland athletics":      {"lat": 37.751, "lon": -122.200, "dome": False},
    "toronto blue jays":      {"lat": 43.641, "lon":  -79.389, "dome": True},
    "atlanta braves":         {"lat": 33.891, "lon":  -84.468, "dome": False},
    "milwaukee brewers":      {"lat": 43.028, "lon":  -87.971, "dome": True},
    "st. louis cardinals":    {"lat": 38.623, "lon":  -90.193, "dome": False},
    "chicago cubs":           {"lat": 41.948, "lon":  -87.655, "dome": False},
    "arizona diamondbacks":   {"lat": 33.445, "lon": -112.067, "dome": True},
    "los angeles dodgers":    {"lat": 34.073, "lon": -118.240, "dome": False},
    "san francisco giants":   {"lat": 37.778, "lon": -122.389, "dome": False},
    "cleveland guardians":    {"lat": 41.496, "lon":  -81.685, "dome": False},
    "seattle mariners":       {"lat": 47.591, "lon": -122.332, "dome": True},
    "miami marlins":          {"lat": 25.778, "lon":  -80.220, "dome": True},
    "new york mets":          {"lat": 40.757, "lon":  -73.846, "dome": False},
    "washington nationals":   {"lat": 38.873, "lon":  -77.008, "dome": False},
    "baltimore orioles":      {"lat": 39.284, "lon":  -76.622, "dome": False},
    "san diego padres":       {"lat": 32.707, "lon": -117.157, "dome": False},
    "philadelphia phillies":  {"lat": 39.906, "lon":  -75.166, "dome": False},
    "pittsburgh pirates":     {"lat": 40.447, "lon":  -80.006, "dome": False},
    "texas rangers":          {"lat": 32.748, "lon":  -97.083, "dome": True},
    "tampa bay rays":         {"lat": 27.768, "lon":  -82.653, "dome": True},
    "boston red sox":         {"lat": 42.347, "lon":  -71.097, "dome": False},
    "cincinnati reds":        {"lat": 39.097, "lon":  -84.507, "dome": False},
    "colorado rockies":       {"lat": 39.756, "lon": -104.994, "dome": False},
    "kansas city royals":     {"lat": 39.051, "lon":  -94.480, "dome": False},
    "detroit tigers":         {"lat": 42.339, "lon":  -83.049, "dome": False},
    "minnesota twins":        {"lat": 44.982, "lon":  -93.278, "dome": False},
    "chicago white sox":      {"lat": 41.830, "lon":  -87.634, "dome": False},
    "new york yankees":       {"lat": 40.829, "lon":  -73.927, "dome": False},
}

# ── CF bearings (compass degrees from home plate toward CF) ───────────────────
CF_BEARING = {
    "los angeles angels":     225,
    "baltimore orioles":       90,
    "boston red sox":          95,
    "cleveland guardians":    225,
    "detroit tigers":         135,
    "kansas city royals":      45,
    "minnesota twins":        315,
    "chicago white sox":      315,
    "washington nationals":   315,
    "atlanta braves":          25,
    "new york mets":          335,
    "philadelphia phillies":  330,
    "chicago cubs":           350,
    "cincinnati reds":        315,
    "pittsburgh pirates":     315,
    "st. louis cardinals":      5,
    "colorado rockies":       330,
    "los angeles dodgers":     25,
    "san diego padres":       315,
    "san francisco giants":    20,
    "new york yankees":       315,
    "oakland athletics":      315,
}


def _fetch_weather(lat: float, lon: float, date_str: str) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation,windspeed_10m,winddirection_10m,relativehumidity_2m"
        f"&temperature_unit=fahrenheit&windspeed_unit=mph"
        f"&timezone=auto&start_date={date_str}&end_date={date_str}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        h = resp.json().get("hourly", {})
        idx = 19  # 7 PM local
        return {
            "temp_f":    h.get("temperature_2m",  [None]*24)[idx],
            "wind_mph":  h.get("windspeed_10m",   [None]*24)[idx],
            "wind_dir":  h.get("winddirection_10m",[None]*24)[idx],
            "precip_mm": h.get("precipitation",   [None]*24)[idx],
            "humidity":  h.get("relativehumidity_2m",[None]*24)[idx],
        }
    except Exception as e:
        print(f"  [weather] API error: {e}")
        return {"temp_f": None, "wind_mph": None, "wind_dir": None,
                "precip_mm": None, "humidity": None}


def _temp_factor(temp_f) -> float:
    if temp_f is None:
        return 1.0
    return 1 + 0.0004 * (temp_f - 70)


def _wind_factor(wind_mph, wind_dir, team: str) -> float:
    if wind_mph is None or wind_dir is None:
        return 1.0
    bearing = CF_BEARING.get(clean_name(team))
    if bearing is None:
        return 1.0
    diff = abs((wind_dir - bearing + 360) % 360)
    if diff > 180:
        diff = 360 - diff
    if diff < 60:
        return 1 + 0.005 * min(wind_mph, 25)   # tailwind
    elif diff > 120:
        return 1 - 0.004 * min(wind_mph, 25)   # headwind
    return 1.0


def _precip_factor(precip_mm) -> float:
    if precip_mm is None or precip_mm == 0:
        return 1.0
    return 1 - 0.03 * min(precip_mm, 5)


def get_weather_data(today_games: pd.DataFrame, today: date = None) -> pd.DataFrame:
    if today is None:
        today = date.today()

    date_str = str(today)
    print("[weather] fetching game-time conditions ...")

    rows = []
    for _, game in today_games.iterrows():
        home = game["home_team"]
        pk   = game["game_pk"]

        coords = STADIUM_COORDS.get(clean_name(home))
        if coords is None:
            # try fuzzy
            for key, val in STADIUM_COORDS.items():
                if key in home or home in key:
                    coords = val
                    break

        if coords is None:
            print(f"  [weather] no coords for {home} – neutral")
            rows.append({"game_pk": pk, "home_team": home, "temp_f": None,
                         "wind_mph": None, "wind_dir": None, "precip_mm": None,
                         "dome": False, "weather_mult": 1.0, "weather_label": "unknown"})
            continue

        if coords["dome"]:
            print(f"  [weather] {home} plays in a dome – neutral")
            rows.append({"game_pk": pk, "home_team": home, "temp_f": None,
                         "wind_mph": None, "wind_dir": None, "precip_mm": None,
                         "dome": True, "weather_mult": 1.0, "weather_label": "dome"})
            continue

        w = _fetch_weather(coords["lat"], coords["lon"], date_str)
        mult = (_temp_factor(w["temp_f"]) *
                _wind_factor(w["wind_mph"], w["wind_dir"], home) *
                _precip_factor(w["precip_mm"]))

        # Cap weather multiplier — rain shouldn't cause > 15% suppression
        mult = max(0.85, min(1.15, mult))

        # Build label — include rain warning if significant
        if w["temp_f"] is not None:
            label = f"{w['temp_f']:.0f}°F, wind {w['wind_mph']:.0f} mph"
            if w["precip_mm"] is not None and w["precip_mm"] > 1.0:
                label += f", rain {w['precip_mm']:.1f}mm"
        else:
            label = "unknown"

        print(f"  [weather] {home} | {label} | mult {mult:.3f}")
        rows.append({"game_pk": pk, "home_team": home,
                     "temp_f": w["temp_f"], "wind_mph": w["wind_mph"],
                     "wind_dir": w["wind_dir"], "precip_mm": w["precip_mm"],
                     "dome": False, "weather_mult": mult, "weather_label": label})

    return pd.DataFrame(rows)
