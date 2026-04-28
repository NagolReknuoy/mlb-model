# =============================================================================
# batting.py — Team offensive metrics via MLB Stats API
#
# Uses the same statsapi.mlb.com endpoint as schedule.py and pitching.py
# so it never gets blocked. Falls back to prior season if current is thin.
# =============================================================================

import requests
import pandas as pd
import numpy as np
from datetime import date
from modules.utils import clean_name

import warnings
warnings.filterwarnings("ignore")


def _get_mlb_team_batting(season: int) -> pd.DataFrame:
    """Pull team batting stats from MLB Stats API."""
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


def get_batting_data(today_games: pd.DataFrame, season: int = None) -> dict:
    if season is None:
        season = date.today().year

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

    # Derived metrics
    tb = raw.copy()
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

    # Offensive multiplier capped at ±15%
    def off_mult(woba):
        if pd.isna(woba) or lg["lg_woba"] == 0:
            return 1.0
        return float(np.clip((woba / lg["lg_woba"]) ** 0.8, 0.85, 1.15))

    tb["off_mult"] = tb["woba"].apply(off_mult)

    print(f"[batting] {len(tb)} teams | "
          f"lg wOBA {lg['lg_woba']:.3f} | "
          f"lg OPS {lg['lg_ops']:.3f} | "
          f"lg K% {lg['lg_k_pct']*100:.1f}%")

    return {"team_batting": tb, "league_batting": lg}
