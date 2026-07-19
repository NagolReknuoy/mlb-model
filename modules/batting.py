# =============================================================================
# batting.py — Team offensive metrics via MLB Stats API
#
# Uses the same statsapi.mlb.com endpoint as schedule.py and pitching.py
# so it never gets blocked. Falls back to prior season if current is thin.
#
# FIX (backtest lookahead leakage): the MLB Stats API's "season" batting
# endpoint always returns CUMULATIVE stats as of whenever you call it — not
# as of any particular date. That meant a backtest for e.g. April 5th was
# scoring the model using each team's full-season batting line (including
# games that happened months after April 5th), inflating backtest accuracy.
#
# There's no "as of date" version of this endpoint, so the fix is a daily
# snapshot: every time run_model.py runs for TODAY, it saves the current
# team batting stats to data/batting_snapshots/<date>.csv. From then on,
# backtesting that date reads the snapshot instead of hitting the live
# endpoint, so it sees exactly what was true on that day — no leakage.
#
# IMPORTANT: this only fixes backtests going forward. Any date before you
# start saving snapshots has no snapshot to fall back on, so batting.py
# will print a loud warning and fall back to (leaky) current stats for
# those older dates. Once you've run daily.py for a few weeks you'll have
# a real, leak-free backtest window to trust.
# =============================================================================

import os
import requests
import pandas as pd
import numpy as np
from datetime import date
from modules.utils import clean_name

import warnings
warnings.filterwarnings("ignore")

SNAPSHOT_DIR = "data/batting_snapshots"


def _snapshot_path(d: date) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{d}.csv")


def _save_snapshot(team_batting: pd.DataFrame, d: date):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    team_batting.to_csv(_snapshot_path(d), index=False)
    print(f"[batting] saved snapshot -> {_snapshot_path(d)}")


def _load_snapshot(d: date):
    path = _snapshot_path(d)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _get_mlb_team_batting(season: int) -> pd.DataFrame:
    """Pull team batting stats from MLB Stats API. Always CURRENT cumulative
    stats as of right now — see the leakage warning at the top of this file."""
    url = (
        f"https://statsapi.mlb.com/api/v1/teams/stats"
        f"?season={season}&sportId=1&stats=season&group=hitting"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return pd.DataFrame()

        rows = []
        for s in splits:
            st = s.get("stat", {})
            rows.append({
                "team":    clean_name(s.get("team", {}).get("name", "")),
                "pa":      float(st.get("plateAppearances", 0) or 0),
                "ab":      float(st.get("atBats", 0) or 0),
                "hits":    float(st.get("hits", 0) or 0),
                "doubles": float(st.get("doubles", 0) or 0),
                "triples": float(st.get("triples", 0) or 0),
                "hr":      float(st.get("homeRuns", 0) or 0),
                "bb":      float(st.get("baseOnBalls", 0) or 0),
                "hbp":     float(st.get("hitByPitch", 0) or 0),
                "sf":      float(st.get("sacFlies", 0) or 0),
                "k":       float(st.get("strikeOuts", 0) or 0),
                "runs":    float(st.get("runs", 0) or 0),
                "avg":     float(st.get("avg", 0) or 0),
                "obp":     float(st.get("obp", 0) or 0),
                "slg":     float(st.get("slg", 0) or 0),
                "ops":     float(st.get("ops", 0) or 0),
            })
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"[batting] MLB Stats API error: {e}")
        return pd.DataFrame()


def _compute_woba(df: pd.DataFrame) -> pd.Series:
    """
    Compute wOBA from counting stats using standard 2025 weights.
    wOBA = (0.69*BB + 0.72*HBP + 0.89*1B + 1.27*2B + 1.62*3B + 2.10*HR)
           / (AB + BB + SF + HBP)
    """
    singles = df["hits"] - df["doubles"] - df["triples"] - df["hr"]
    num = (0.69 * df["bb"]  +
           0.72 * df["hbp"] +
           0.89 * singles   +
           1.27 * df["doubles"] +
           1.62 * df["triples"] +
           2.10 * df["hr"])
    denom = (df["ab"] + df["bb"] + df["sf"] + df["hbp"]).clip(lower=1)
    return (num / denom).round(3)


def _derive(tb: pd.DataFrame) -> dict:
    """Compute wOBA/K%/BB%/ISO/off_mult + league averages from a raw batting df."""
    tb = tb.copy()
    tb["woba"]  = _compute_woba(tb)
    tb["k_pct"] = (tb["k"]  / tb["pa"].clip(lower=1)).round(3)
    tb["bb_pct"]= (tb["bb"] / tb["pa"].clip(lower=1)).round(3)
    tb["iso"]   = (tb["slg"] - tb["avg"]).round(3)

    lg = {
        "lg_woba":   round(float(tb["woba"].mean()),  3),
        "lg_ops":    round(float(tb["ops"].mean()),   3),
        "lg_k_pct":  round(float(tb["k_pct"].mean()), 3),
        "lg_bb_pct": round(float(tb["bb_pct"].mean()),3),
        "lg_iso":    round(float(tb["iso"].mean()),   3),
    }

    def off_mult(woba):
        if pd.isna(woba) or lg["lg_woba"] == 0:
            return 1.0
        return float(np.clip((woba / lg["lg_woba"]) ** 0.8, 0.85, 1.15))

    tb["off_mult"] = tb["woba"].apply(off_mult)
    return tb, lg


def get_batting_data(today_games: pd.DataFrame, season: int = None,
                     target_date: date = None) -> dict:
    if season is None:
        season = date.today().year
    if target_date is None:
        target_date = date.today()

    is_today = (target_date == date.today())

    # ── Backtesting a past date: use the saved snapshot if we have one ───────
    if not is_today:
        snap = _load_snapshot(target_date)
        if snap is not None:
            print(f"[batting] using saved snapshot for {target_date} (leak-free)")
            tb, lg = _derive(snap)
            print(f"[batting] {len(tb)} teams | lg wOBA {lg['lg_woba']:.3f}")
            return {"team_batting": tb, "league_batting": lg}
        else:
            print(f"[batting] ⚠️  NO SNAPSHOT for {target_date} — falling back to "
                  f"CURRENT cumulative stats. This WILL leak future data into "
                  f"this date's prediction/backtest. Run daily.py going forward "
                  f"to build up a leak-free snapshot history.")

    print(f"[batting] fetching MLB Stats API team batting for {season} ...")
    raw = _get_mlb_team_batting(season)

    # Blend with prior season if current season is still thin
    prev_season = season - 1
    raw_prev = _get_mlb_team_batting(prev_season)

    if not raw.empty and not raw_prev.empty:
        print(f"[batting] blending {season}(2x) + {prev_season}(1x) for stability ...")
        # Weight current season 2x by stacking it twice before averaging
        numeric_cols = [c for c in raw.columns if c != "team"]
        combined = pd.concat([raw, raw, raw_prev], ignore_index=True)
        raw = combined.groupby("team")[numeric_cols].mean().reset_index()
    elif raw.empty and not raw_prev.empty:
        print(f"[batting] {season} unavailable – using {prev_season} as proxy")
        raw = raw_prev
    elif raw.empty:
        print("[batting] no data available – using neutral batting")
        return {"team_batting": pd.DataFrame(), "league_batting": {}}

    tb, lg = _derive(raw)

    print(f"[batting] {len(tb)} teams | "
          f"lg wOBA {lg['lg_woba']:.3f} | "
          f"lg OPS {lg['lg_ops']:.3f} | "
          f"lg K% {lg['lg_k_pct']*100:.1f}%")

    # Only persist a snapshot when this really is "today" — never write a
    # snapshot while re-running an old date, or you'd bake the leak in.
    if is_today:
        _save_snapshot(raw, target_date)

    return {"team_batting": tb, "league_batting": lg}
