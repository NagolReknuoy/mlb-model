# run_model.py — Main 
# Usage:
#   python run_model.py                  # today's games
#   python run_model.py --date 2026-04-21
#   python run_model.py --csv            # also saves results/predictions_DATE.csv

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import date, datetime

# Make sure modules folder is found
sys.path.insert(0, os.path.dirname(__file__))

from modules.utils     import simulate_game, clean_name
from modules.schedule  import load_season_data
from modules.pitching  import get_pitching_data, pitch_factor, starter_ip, blend_starter_bullpen
from modules.batting   import get_batting_data
from modules.parks     import get_park_data, park_run_mult, park_info
from modules.weather   import get_weather_data
from modules.trends    import get_trends
from modules.odds      import fetch_odds, find_value_bets, print_value_bets


def run_model(today: date = None, output_csv: bool = False, fetch_odds_data: bool = True):
    if today is None:
        today = date.today()

    print(f"\n{'='*60}")
    print(f"  MLB Run Model | {today}")
    print(f"{'='*60}\n")

    # ── 1. Schedule + team strength ──────────────────────────────────────────
    sched = load_season_data(today)
    if sched["today_games"].empty:
        print("No games scheduled today.")
        return None

    # ── 2. Pitching ──────────────────────────────────────────────────────────
    pitch = get_pitching_data(sched["today_games"], today)

    # ── 3. Batting ───────────────────────────────────────────────────────────
    bat = get_batting_data(sched["today_games"], season=today.year)

    # ── 4. Parks ─────────────────────────────────────────────────────────────
    get_park_data()  # prints confirmation

    # ── 5. Weather ───────────────────────────────────────────────────────────
    wx = get_weather_data(sched["today_games"], today)

    # ── 6. Trends ─────────────────────────────────────────────────────────────
    tr = get_trends(
        sched["past_games"], sched["today_games"],
        today=today, league_avg=sched["league_avg_runs"]
    )

    # ── 7. Predictions ────────────────────────────────────────────────────────
    print("\n[model] running predictions ...")
    lg   = sched["league_avg_runs"]
    hfm  = sched["home_field_mult"]
    afm  = sched["away_field_mult"]
    lp   = pitch["league_pitch"]
    pdf  = pitch["pitchers_df"]
    tb   = bat["team_batting"]

    results = []
    for _, game in sched["today_games"].iterrows():
        home    = game["home_team"]
        away    = game["away_team"]
        pk      = game["game_pk"]

        # Team strength
        h_str = sched["team_stats"][sched["team_stats"]["team"] == home]
        a_str = sched["team_stats"][sched["team_stats"]["team"] == away]
        if h_str.empty or a_str.empty:
            print(f"  [model] skipping {home} vs {away} – no team stats")
            continue
        h_str = h_str.iloc[0]
        a_str = a_str.iloc[0]

        # Use geometric mean to prevent extreme OS*DS compounding
        # sqrt(OS * DS) dampens outlier interactions while preserving direction
        # e.g. great offense vs terrible defense: sqrt(1.25*1.35)=1.30 not 1.69
        base_home = lg * np.sqrt(h_str["OS"] * a_str["DS"]) * hfm
        base_away = lg * np.sqrt(a_str["OS"] * h_str["DS"]) * afm

        # Park factor (annual × monthly)
        pm = park_run_mult(home, today)
        base_home *= pm
        base_away *= pm

        # Batting multiplier
        def get_bat_mult(team):
            if tb.empty:
                return 1.0
            row = tb[tb["team"] == team]
            if row.empty:
                # fuzzy match
                for t in tb["team"]:
                    if t in team or team in t:
                        row = tb[tb["team"] == t]
                        break
            if row.empty or pd.isna(row.iloc[0]["off_mult"]):
                return 1.0
            return float(row.iloc[0]["off_mult"])

        base_home *= get_bat_mult(home)
        base_away *= get_bat_mult(away)

        # Pitching multiplier
        p_row = pdf[pdf["game_pk"] == pk]
        if not p_row.empty:
            p = p_row.iloc[0]
            home_pitch_mult = blend_starter_bullpen(
                pitch_factor(p.get("away_velo"), p.get("away_kbb"), None,
                             lp.get("lg_velo"), lp.get("lg_kbb"),
                             era_proxy=p.get("away_era"), lg_era=lp.get("lg_era")),
                starter_ip(p.get("away_ppg"))
            )
            away_pitch_mult = blend_starter_bullpen(
                pitch_factor(p.get("home_velo"), p.get("home_kbb"), None,
                             lp.get("lg_velo"), lp.get("lg_kbb"),
                             era_proxy=p.get("home_era"), lg_era=lp.get("lg_era")),
                starter_ip(p.get("home_ppg"))
            )
        else:
            home_pitch_mult = away_pitch_mult = 1.0

        base_home *= home_pitch_mult
        base_away *= away_pitch_mult

        # Weather
        wx_row = wx[wx["game_pk"] == pk]
        wx_mult  = float(wx_row.iloc[0]["weather_mult"])  if not wx_row.empty else 1.0
        wx_label = str(wx_row.iloc[0]["weather_label"])   if not wx_row.empty else "unknown"
        base_home *= wx_mult
        base_away *= wx_mult

        # Trends
        tr_row = tr["trends_mult"][tr["trends_mult"]["game_pk"] == pk]
        if not tr_row.empty:
            base_home *= float(tr_row.iloc[0]["home_trend_mult"])
            base_away *= float(tr_row.iloc[0]["away_trend_mult"])

        # Simulate
        sim = simulate_game(base_home, base_away)

        # Park info
        pi      = park_info(home)
        pk_type = pi.get("type", "unknown")
        pk_note = pi.get("note", "")

        # H2H
        h2h_row = tr["h2h"][tr["h2h"]["game_pk"] == pk]
        h2h_str = str(h2h_row.iloc[0]["h2h_str"]) if not h2h_row.empty else "no prior meetings"

        # Pitcher names
        home_sp = str(p_row.iloc[0]["home_pitcher"]).title() if not p_row.empty and p_row.iloc[0]["home_pitcher"] else "TBD"
        away_sp = str(p_row.iloc[0]["away_pitcher"]).title() if not p_row.empty and p_row.iloc[0]["away_pitcher"] else "TBD"

        # Label doubleheader games
        dh_info = game.get("doubleheader", "N")
        gm_num  = game.get("game_num", 1)
        dh_tag  = f" (G{gm_num})" if dh_info != "N" else ""

        results.append({
            "Home":         home.title() + dh_tag,
            "Away":         away.title(),
            "Home_SP":      home_sp,
            "Away_SP":      away_sp,
            "Home_Win_Pct": sim["home_win"],
            "Away_Win_Pct": sim["away_win"],
            "xTotal_Runs":  sim["total"],
            "Park_Factor":  round(pm, 3),
            "Park_Type":    pk_type,
            "Park_Note":    pk_note,
            "Weather":      wx_label,
            "Wx_Mult":      round(wx_mult, 3),
            "H2H":          h2h_str,
            "Home_lambda":  round(base_home, 3),
            "Away_lambda":  round(base_away, 3),
        })

    if not results:
        print("No predictions generated.")
        return None

    df = pd.DataFrame(results)
    _print_results(df)

    # ── 8. Odds + value bets ─────────────────────────────────────────────────
    if fetch_odds_data:
        print("\n[odds] fetching sportsbook lines ...")
        odds_df = fetch_odds(today)
        if not odds_df.empty:
            bets = find_value_bets(df, odds_df)
            print_value_bets(bets)
            if output_csv and not bets.empty:
                bets_path = f"results/value_bets_{today}.csv"
                bets.to_csv(bets_path, index=False)
                print(f"[odds] value bets saved → {bets_path}")
    else:
        print("\n[odds] skipped (--no-odds flag set)")

    if output_csv:
        os.makedirs("results", exist_ok=True)
        path = f"results/predictions_{today}.csv"
        df.to_csv(path, index=False)
        print(f"\n[model] saved → {path}")

    return df


def _print_results(df: pd.DataFrame):
    # Column widths
    W_GAME  = 34
    W_PCT   = 6
    W_RUNS  = 6
    W_PARK  = 14
    W_WX    = 22
    W_H2H   = 18
    W_SP    = 32

    total = W_GAME + W_PCT*2 + W_RUNS + W_PARK + W_WX + W_H2H + W_SP + 10
    bar   = "=" * total

    print(f"\n{bar}")
    print(f"  MLB PREDICTIONS")
    print(f"{bar}")

    # Header row 1 — matchup info
    print(f"  {'MATCHUP':<{W_GAME}} {'H%':>{W_PCT}} {'A%':>{W_PCT}} "
          f"{'xRuns':>{W_RUNS}}  {'PARK':<{W_PARK}} {'WEATHER':<{W_WX}} "
          f"{'H2H':<{W_H2H}}")
    # Header row 2 — pitchers
    print(f"  {'STARTING PITCHERS':<{W_GAME+W_PCT*2+W_RUNS+3}}"
          f"{'PARK NOTE'}")
    print(f"{'-' * total}")

    for _, r in df.iterrows():
        matchup   = f"{r['Home']} vs {r['Away']}"
        park      = f"{r['Park_Type']}×{r['Park_Factor']:.2f}"
        sps       = f"{r['Home_SP']} vs {r['Away_SP']}"
        note      = r['Park_Note'] if r['Park_Note'] else ""
        wx        = r['Weather'][:W_WX]

        # Truncate long names
        if len(matchup) > W_GAME:
            matchup = matchup[:W_GAME-1]
        if len(sps) > W_SP:
            sps = sps[:W_SP-1]

        print(f"  {matchup:<{W_GAME}} {r['Home_Win_Pct']:>{W_PCT-1}.1f}% "
              f"{r['Away_Win_Pct']:>{W_PCT-1}.1f}% "
              f"{r['xTotal_Runs']:>{W_RUNS}.2f}  {park:<{W_PARK}} "
              f"{wx:<{W_WX}} {r['H2H']:<{W_H2H}}")
        print(f"  {sps:<{W_GAME+W_PCT*2+W_RUNS+3}}{note}")
        print(f"{'-' * total}")

    print(f"{bar}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Run Model")
    parser.add_argument("--date", type=str, default=None,
                        help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--csv",  action="store_true",
                        help="Save results to CSV")
    parser.add_argument("--no-odds", action="store_true",
                        help="Skip odds API fetch (saves credits while testing)")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    run_model(today=run_date, output_csv=args.csv, fetch_odds_data=not args.no_odds)
