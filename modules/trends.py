# =============================================================================
# trends.py — Recent form, win streaks, head-to-head records
# =============================================================================

import pandas as pd
import numpy as np
from datetime import date, timedelta
from modules.utils import clean_name


def get_trends(past_games: pd.DataFrame, today_games: pd.DataFrame,
               today: date = None, lookback: int = 10,
               league_avg: float = None) -> dict:
    if today is None:
        today = date.today()

    print(f"[trends] computing team form and head-to-head (lookback {lookback} days) ...")

    # Build per-team game log from both home and away perspective
    home_log = past_games[["date","home_team","home_runs","away_runs"]].copy()
    home_log = home_log.rename(columns={"home_team":"team","home_runs":"rs","away_runs":"ra"})
    home_log["win"] = home_log["rs"] > home_log["ra"]

    away_log = past_games[["date","away_team","away_runs","home_runs"]].copy()
    away_log = away_log.rename(columns={"away_team":"team","away_runs":"rs","home_runs":"ra"})
    away_log["win"] = away_log["rs"] > away_log["ra"]

    game_log = pd.concat([home_log, away_log], ignore_index=True)
    game_log = game_log.sort_values(["team","date"]).reset_index(drop=True)

    # ── Last-N games form per team ────────────────────────────────────────────
    # Use a loop instead of groupby().apply() to avoid pandas version issues
    teams = game_log["team"].unique()
    form_rows = []
    streak_rows = []

    for team in teams:
        grp = game_log[game_log["team"] == team].sort_values("date")
        last = grp.tail(lookback)

        # Form stats
        g      = len(last)
        rs     = last["rs"].sum()
        ra     = last["ra"].sum()
        wins   = last["win"].sum()
        rpg    = rs / max(g, 1)
        rapg   = ra / max(g, 1)
        form_rows.append({
            "team":      team,
            "last_rs":   rs,
            "last_ra":   ra,
            "last_wins": wins,
            "last_g":    g,
            "rpg":       rpg,
            "rapg":      rapg,
        })

        # Streak
        results = last["win"].tolist()
        if results:
            last_result = results[-1]
            count = 0
            for r in reversed(results):
                if r == last_result:
                    count += 1
                else:
                    break
            streak_type = "W" if last_result else "L"
        else:
            streak_type = "W"
            count = 0
        streak_rows.append({
            "team":        team,
            "streak_type": streak_type,
            "streak_n":    count,
            "streak_mult": 1 + 0.003 * min(count, 5) if streak_type == "W"
                           else 1 - 0.003 * min(count, 5)
        })

    form    = pd.DataFrame(form_rows)
    streaks = pd.DataFrame(streak_rows)
    form    = form.merge(streaks, on="team", how="left")

    # Normalise vs league avg
    lg = league_avg or form["rpg"].mean()
    form["form_off_mult"] = (form["rpg"]  / lg).clip(0.7, 1.3)
    form["form_def_mult"] = (form["rapg"] / lg).clip(0.7, 1.3)

    # ── Head-to-head ──────────────────────────────────────────────────────────
    h2h_rows = []
    for _, game in today_games.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        pk   = game["game_pk"]

        matchups = past_games[
            ((past_games["home_team"] == home) & (past_games["away_team"] == away)) |
            ((past_games["home_team"] == away) & (past_games["away_team"] == home))
        ]

        home_wins = int((
            ((matchups["home_team"] == home) & (matchups["home_runs"] > matchups["away_runs"])) |
            ((matchups["away_team"] == home) & (matchups["away_runs"] > matchups["home_runs"]))
        ).sum())

        away_wins = int((
            ((matchups["home_team"] == away) & (matchups["home_runs"] > matchups["away_runs"])) |
            ((matchups["away_team"] == away) & (matchups["away_runs"] > matchups["home_runs"]))
        ).sum())

        total   = len(matchups)
        h2h_str = f"{home_wins}-{away_wins} (home-away)" if total > 0 else "no prior meetings"
        h2h_rows.append({"game_pk": pk, "home_team": home, "away_team": away, "h2h_str": h2h_str})

    h2h = pd.DataFrame(h2h_rows)

    # ── Trend multiplier per game ─────────────────────────────────────────────
    def get_form_val(team, col, default=1.0):
        row = form[form["team"] == team]
        if row.empty:
            return default
        val = row.iloc[0][col]
        return float(val) if not pd.isna(val) else default

    trend_rows = []
    for _, game in today_games.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        h_off = get_form_val(home, "form_off_mult")
        h_def = get_form_val(home, "form_def_mult")
        h_str = get_form_val(home, "streak_mult")
        a_off = get_form_val(away, "form_off_mult")
        a_def = get_form_val(away, "form_def_mult")
        a_str = get_form_val(away, "streak_mult")
        trend_rows.append({
            "game_pk":         game["game_pk"],
            "home_trend_mult": h_off * a_def * h_str,
            "away_trend_mult": a_off * h_def * a_str,
        })

    trends_mult = pd.DataFrame(trend_rows)

    print(f"[trends] {len(form)} teams computed")
    return {"team_form": form, "h2h": h2h, "trends_mult": trends_mult}
