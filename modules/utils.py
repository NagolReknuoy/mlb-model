# utils.py — Shared helpers used by all modules

import numpy as np
from datetime import date


def clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.lower().strip()


def safe_mean(values, fallback=1.0):
    arr = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(arr)) if arr else fallback


def simulate_game(lambda_home: float, lambda_away: float, sims: int = 10_000) -> dict:
    """
    Poisson simulation of a baseball game.

    Approach: clamp each lambda independently to 3.5-5.5.
    League average is 4.49 R/team/game.
    3.5 = weak offense/great pitching environment
    5.5 = strong offense/weak pitching environment
    """
    lh = max(3.5, min(5.5, lambda_home))
    la = max(3.5, min(5.5, lambda_away))

    h = np.random.poisson(lh, sims)
    a = np.random.poisson(la, sims)

    home_wins = float(np.mean(h > a))
    away_wins = float(np.mean(a > h))
    ties      = float(np.mean(h == a))

    # Redistribute ties
    total_decided = home_wins + away_wins
    if total_decided > 0:
        home_wins += ties * (home_wins / total_decided)
        away_wins += ties * (away_wins / total_decided)

    return {
        "home_win": round(home_wins * 100, 1),
        "away_win": round(away_wins * 100, 1),
        "total":    round(float(np.mean(h + a)), 2),
    }


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def current_season() -> int:
    return date.today().year
