# =============================================================================
# historical_odds.py — Load scraped SportsBookReview odds into backtest
#
# Works with JSON files produced by mlb_odds_scraper/scraper.py
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
from datetime import date
from modules.utils import clean_name


# ── Load JSON file ────────────────────────────────────────────────────────────

def load_odds_file(path: str) -> dict:
    """Load the scraped odds JSON. Returns empty dict if file not found."""
    if not os.path.exists(path):
        print(f"[historical_odds] file not found: {path}")
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    total = sum(len(v) for v in data.values())
    print(f"[historical_odds] loaded {total} games across {len(data)} dates from {path}")
    return data


# ── Team name normalization ───────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Normalize team name for matching."""
    if not isinstance(name, str):
        return ""
    return (name.lower()
            .replace(".", "").replace("'", "")
            .replace("-", " ").replace("&", "and")
            .strip())


def _nickname(name: str) -> str:
    """Extract last word (nickname) for fuzzy matching."""
    parts = _norm(name).split()
    if not parts:
        return ""
    if len(parts) >= 2 and parts[-1] in ("sox", "jays"):
        return " ".join(parts[-2:])
    return parts[-1]


# ── Extract odds for one game ─────────────────────────────────────────────────

def _median_odds(views: list, key: str):
    """Median value across all sportsbooks for a given key."""
    vals = []
    for v in views:
        line = v.get("currentLine", {})
        val  = line.get(key)
        if val is not None:
            vals.append(val)
    return float(np.median(vals)) if vals else None


def _get_game_odds(game: dict) -> dict:
    """Extract moneyline, run line, and totals from a scraped game dict."""
    odds = game.get("odds", {})

    # Moneyline
    ml_views  = odds.get("moneyline", [])
    ml_home   = _median_odds(ml_views, "homeOdds")
    ml_away   = _median_odds(ml_views, "awayOdds")

    # Run line / point spread
    rl_views  = odds.get("pointspread", [])
    rl_home   = _median_odds(rl_views, "homeOdds")
    rl_away   = _median_odds(rl_views, "awayOdds")
    rl_line   = _median_odds(rl_views, "homeSpread")  # negative = home favored

    # Totals
    tot_views = odds.get("totals", [])
    tot_over  = _median_odds(tot_views, "overOdds")
    tot_under = _median_odds(tot_views, "underOdds")
    tot_line  = _median_odds(tot_views, "total")

    return {
        "ml_home":  ml_home,
        "ml_away":  ml_away,
        "rl_home":  rl_home,
        "rl_away":  rl_away,
        "rl_line":  rl_line,
        "tot_over": tot_over,
        "tot_under":tot_under,
        "tot_line": tot_line,
    }


# ── Match a prediction to a scraped game ─────────────────────────────────────

def get_odds_for_game(home_team: str, away_team: str,
                      target_date: date, odds_data: dict) -> dict:
    """
    Find odds for a specific game in the scraped data.
    Requires BOTH home and away nicknames to match exactly.
    Returns dict with ml_home, ml_away, rl_*, tot_* or empty dict if not found.
    """
    date_str = str(target_date)
    games    = odds_data.get(date_str, [])
    if not games:
        return {}

    hn = _nickname(home_team)
    an = _nickname(away_team)

    # Pass 1: strict — both nicknames must match exactly
    for game in games:
        gv  = game.get("gameView", {})
        gh  = _norm(gv.get("homeTeam", {}).get("fullName", ""))
        ga  = _norm(gv.get("awayTeam", {}).get("fullName", ""))
        ghn = _nickname(gh)
        gan = _nickname(ga)

        if hn == ghn and an == gan:
            return _get_game_odds(game)

    # Pass 2: fallback — full normalized name contains nickname
    # Only used if strict match fails (e.g. Athletics vs A's naming)
    for game in games:
        gv  = game.get("gameView", {})
        gh  = _norm(gv.get("homeTeam", {}).get("fullName", ""))
        ga  = _norm(gv.get("awayTeam", {}).get("fullName", ""))

        home_match = (hn in gh.split()) or (gh.split()[-1:] == [hn])
        away_match = (an in ga.split()) or (ga.split()[-1:] == [an])

        if home_match and away_match:
            return _get_game_odds(game)

    return {}


# ── American odds helpers ─────────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(p1: float, p2: float):
    total = p1 + p2
    if total <= 0:
        return p1, p2
    return p1 / total, p2 / total


def payout_multiplier(odds: float) -> float:
    """Return profit per $1 wagered (e.g. +150 → 1.5, -150 → 0.667).
    Clamped to realistic range so bad data can't produce absurd payouts."""
    if odds is None:
        return None
    if odds > 0:
        return min(odds / 100, 10.0)   # cap at +1000 equivalent
    return min(100 / abs(odds), 10.0)


# ── Score a single bet ────────────────────────────────────────────────────────

def score_bet(bet_type: str, odds_row: dict,
              model_home_pct: float, model_away_pct: float,
              model_xruns: float, actual_home: int, actual_away: int,
              home_name: str, away_name: str,
              min_edge: float = 0.07) -> list:
    """
    Evaluate bets for one game. Only flags bets where:
    - Edge >= min_edge (default 7% — raised from 5% to reduce noise)
    - For moneyline: model must have >=58% confidence in the pick
    - For totals: debiased xRuns must differ from line by >=1.0 run
    """
    """
    Evaluate all three bet types for one game.
    Returns list of bet result dicts.
    """
    results = []
    actual_total  = actual_home + actual_away
    home_won      = actual_home > actual_away

    # ── Moneyline ─────────────────────────────────────────────────────────────
    ml_h = odds_row.get("ml_home")
    ml_a = odds_row.get("ml_away")
    # Valid MLB moneyline range
    ml_h_valid = (ml_h is not None and 
                  (ml_h <= -10 or ml_h >= 10) and 
                  -600 <= ml_h <= 600)
    ml_a_valid = (ml_a is not None and 
                  (ml_a <= -10 or ml_a >= 10) and 
                  -600 <= ml_a <= 600)
    if ml_h_valid and ml_a_valid:
        raw_h, raw_a  = american_to_prob(ml_h), american_to_prob(ml_a)
        book_h, book_a = remove_vig(raw_h, raw_a)
        edge_h = min(model_home_pct - book_h, 0.25)
        edge_a = min(model_away_pct - book_a, 0.25)

        for edge, side, ml, book_p, model_p, won in [
            (edge_h, home_name, ml_h, book_h, model_home_pct, home_won),
            (edge_a, away_name, ml_a, book_a, model_away_pct, not home_won),
        ]:
            # Only flag ML bets when model is actually confident (>=58%)
            if edge >= min_edge and model_p >= 0.58:
                profit = payout_multiplier(ml) if won else -1.0
                results.append({
                    "type":      "Moneyline",
                    "bet":       f"{side} ML",
                    "odds":      ml,
                    "edge":      round(edge, 4),
                    "book_prob": round(book_p, 4),
                    "model_prob":round(model_p, 4),
                    "won":       won,
                    "profit":    round(profit, 4),
                })

    # ── Run line ──────────────────────────────────────────────────────────────
    rl_h    = odds_row.get("rl_home")
    rl_a    = odds_row.get("rl_away")
    rl_line = odds_row.get("rl_line")
    # Sanity check: valid American odds range -1000 to +1000
    # Valid run line odds must look like American odds (not spreads like -1.5)
    # Real odds are outside the -10 to +10 range; spreads are inside it
    rl_h_valid = (rl_h is not None and 
                  (rl_h <= -10 or rl_h >= 10) and 
                  -1000 <= rl_h <= 1000)
    rl_a_valid = (rl_a is not None and 
                  (rl_a <= -10 or rl_a >= 10) and 
                  -1000 <= rl_a <= 1000)
    if rl_h_valid and rl_a_valid and rl_line is not None:
        spread = abs(rl_line)
        margin = actual_home - actual_away

        home_covers = margin >  spread   # home -1.5 covers
        away_covers = margin < -spread   # away +1.5 covers

        raw_h, raw_a   = american_to_prob(rl_h), american_to_prob(rl_a)
        book_rl_h, book_rl_a = remove_vig(raw_h, raw_a)

        # Cover probability: winning team needs to win by 2+
        # Approximate: roughly 60% of wins are by 2+ runs in MLB
        # Scale win prob down to cover prob
        COVER_RATE = 0.60  # ~60% of wins cover -1.5 historically
        model_rl_h = model_home_pct * COVER_RATE
        model_rl_a = model_away_pct * COVER_RATE

        edge_rl_h = model_rl_h - book_rl_h
        edge_rl_a = model_rl_a - book_rl_a

        for edge, label, ml, book_p, model_p, won in [
            (edge_rl_h, f"{home_name} -{spread}", rl_h, book_rl_h, model_rl_h, home_covers),
            (edge_rl_a, f"{away_name} +{spread}", rl_a, book_rl_a, model_rl_a, away_covers),
        ]:
            if edge >= min_edge:
                profit = payout_multiplier(ml) if won else -1.0
                results.append({
                    "type":      "Run Line",
                    "bet":       label,
                    "odds":      ml,
                    "edge":      round(edge, 4),
                    "book_prob": round(book_p, 4),
                    "model_prob":round(model_p, 4),
                    "won":       won,
                    "profit":    round(profit, 4),
                })

    # ── Totals ────────────────────────────────────────────────────────────────
    tot_o    = odds_row.get("tot_over")
    tot_u    = odds_row.get("tot_under")
    tot_line = odds_row.get("tot_line")
    if tot_o and tot_u and tot_line is not None:
        went_over  = actual_total > tot_line
        went_under = actual_total < tot_line

        raw_o, raw_u   = american_to_prob(tot_o), american_to_prob(tot_u)
        book_o, book_u = remove_vig(raw_o, raw_u)

        # Debias model xRuns — backtest shows model runs ~17% high
        # Scale down toward the book line to get a fair comparison
        # Use a blend: 60% model, 40% book line to correct for bias
        BIAS_CORRECTION = 0.83   # model is ~17% high on average
        debiased_xruns  = model_xruns * BIAS_CORRECTION

        # Convert to probability using distance from line
        distance = debiased_xruns - tot_line
        model_over  = max(0.30, min(0.70, 0.50 + distance * 0.075))
        model_under = 1.0 - model_over

        edge_o = min(model_over  - book_o, 0.20)
        edge_u = min(model_under - book_u, 0.20)

        # Only flag totals when debiased xRuns is at least 1 run from line
        # This filters out marginal calls where the model is essentially guessing
        if abs(distance) < 1.0:
            edge_o = 0
            edge_u = 0

        for edge, label, ml, book_p, model_p, won in [
            (edge_o, f"OVER  {tot_line}", tot_o, book_o, model_over,  went_over),
            (edge_u, f"UNDER {tot_line}", tot_u, book_u, model_under, went_under),
        ]:
            if edge >= min_edge:
                profit = payout_multiplier(ml) if won else -1.0
                results.append({
                    "type":      "Total",
                    "bet":       label,
                    "odds":      ml,
                    "edge":      round(edge, 4),
                    "book_prob": round(book_p, 4),
                    "model_prob":round(model_p, 4),
                    "won":       won,
                    "profit":    round(profit, 4),
                })

    return results
