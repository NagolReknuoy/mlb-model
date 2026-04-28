# =============================================================================
# backtest.py — Back-test the MLB run model against actual results
#
# Usage:
#   python backtest.py --start 2026-04-01 --end 2026-04-24
#   python backtest.py --start 2026-04-01 --end 2026-04-24 --csv
#   python backtest.py --start 2026-04-01 --end 2026-04-24 --game "Yankees"
#   python backtest.py --start 2026-04-01 --end 2026-04-24 --odds mlb_odds_2026.json --csv
# =============================================================================

import argparse
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

sys.path.insert(0, os.path.dirname(__file__))

from modules.utils     import clean_name, simulate_game
from modules.schedule  import load_season_data
from modules.pitching  import get_pitching_data, pitch_factor, starter_ip, blend_starter_bullpen
from modules.batting   import get_batting_data
from modules.parks     import get_park_data, park_run_mult, park_info
from modules.weather   import get_weather_data
from modules.trends    import get_trends
from modules.historical_odds import (load_odds_file, get_odds_for_game,
                                     score_bet)


# ── Fetch actual results ──────────────────────────────────────────────────────

def fetch_actual_results(target_date):
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        "?sportId=1&date=" + str(target_date) + "&gameType=R"
        "&hydrate=team,linescore"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    continue
                teams = game.get("teams", {})
                rows.append({
                    "game_pk":      game["gamePk"],
                    "home_team":    clean_name(teams.get("home", {}).get("team", {}).get("name", "")),
                    "away_team":    clean_name(teams.get("away", {}).get("team", {}).get("name", "")),
                    "home_runs":    teams.get("home", {}).get("score", 0),
                    "away_runs":    teams.get("away", {}).get("score", 0),
                    "doubleheader": game.get("doubleHeader", "N"),
                    "game_num":     game.get("gameNumber", 1),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        print("  [backtest] could not fetch results for " + str(target_date) + ": " + str(e))
        return pd.DataFrame()


# ── Run model silently ────────────────────────────────────────────────────────

def run_model_silent(target_date):
    try:
        sched = load_season_data(target_date)
        if sched["today_games"].empty:
            return pd.DataFrame()

        pitch = get_pitching_data(sched["today_games"], target_date)
        bat   = get_batting_data(sched["today_games"], season=target_date.year)
        get_park_data()
        wx    = get_weather_data(sched["today_games"], target_date)
        tr    = get_trends(sched["past_games"], sched["today_games"],
                           today=target_date, league_avg=sched["league_avg_runs"])

        lg  = sched["league_avg_runs"]
        hfm = sched["home_field_mult"]
        afm = sched["away_field_mult"]
        lp  = pitch["league_pitch"]
        pdf = pitch["pitchers_df"]
        tb  = bat["team_batting"]

        results = []
        for _, game in sched["today_games"].iterrows():
            home = game["home_team"]
            away = game["away_team"]
            pk   = game["game_pk"]

            h_str = sched["team_stats"][sched["team_stats"]["team"] == home]
            a_str = sched["team_stats"][sched["team_stats"]["team"] == away]
            if h_str.empty or a_str.empty:
                continue
            h_str = h_str.iloc[0]
            a_str = a_str.iloc[0]

            base_home = lg * np.sqrt(h_str["OS"] * a_str["DS"]) * hfm
            base_away = lg * np.sqrt(a_str["OS"] * h_str["DS"]) * afm

            pm = park_run_mult(home, target_date)
            base_home *= pm
            base_away *= pm

            def get_bat_mult(team):
                if tb.empty:
                    return 1.0
                row = tb[tb["team"] == team]
                if row.empty:
                    for t in tb["team"]:
                        if t in team or team in t:
                            row = tb[tb["team"] == t]
                            break
                if row.empty or pd.isna(row.iloc[0]["off_mult"]):
                    return 1.0
                return float(row.iloc[0]["off_mult"])

            base_home *= get_bat_mult(home)
            base_away *= get_bat_mult(away)

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

            wx_row  = wx[wx["game_pk"] == pk]
            wx_mult = float(wx_row.iloc[0]["weather_mult"]) if not wx_row.empty else 1.0
            base_home *= wx_mult
            base_away *= wx_mult

            tr_row = tr["trends_mult"][tr["trends_mult"]["game_pk"] == pk]
            if not tr_row.empty:
                base_home *= float(tr_row.iloc[0]["home_trend_mult"])
                base_away *= float(tr_row.iloc[0]["away_trend_mult"])

            sim = simulate_game(base_home, base_away)
            pi  = park_info(home)

            results.append({
                "date":         str(target_date),
                "game_pk":      pk,
                "home_team":    home,
                "away_team":    away,
                "Home":         home.title(),
                "Away":         away.title(),
                "Home_Win_Pct": sim["home_win"],
                "Away_Win_Pct": sim["away_win"],
                "xTotal_Runs":  sim["total"],
                "Home_lambda":  base_home,
                "Away_lambda":  base_away,
                "Park_Type":    pi.get("type", "unknown"),
            })

        return pd.DataFrame(results)

    except Exception as e:
        print("  [backtest] model error for " + str(target_date) + ": " + str(e))
        return pd.DataFrame()


# ── Score one game ────────────────────────────────────────────────────────────

def score_game(pred_row, actual_row):
    home_runs    = int(actual_row["home_runs"])
    away_runs    = int(actual_row["away_runs"])
    actual_total = home_runs + away_runs
    home_won     = home_runs > away_runs

    model_picks_home = pred_row["Home_Win_Pct"] > pred_row["Away_Win_Pct"]
    ml_correct       = (model_picks_home == home_won)

    if pred_row["Home_Win_Pct"] > pred_row["Away_Win_Pct"]:
        fav_margin = home_runs - away_runs
    else:
        fav_margin = away_runs - home_runs
    rl_cover = fav_margin >= 2

    xruns      = pred_row["xTotal_Runs"]
    runs_error = abs(xruns - actual_total)

    game_num = actual_row.get("game_num", 1)
    dh       = actual_row.get("doubleheader", "N")
    dh_label = " (DH-" + str(game_num) + ")" if dh != "N" else ""

    return {
        "date":           pred_row["date"],
        "home":           pred_row["Home"] + dh_label,
        "away":           pred_row["Away"],
        "home_score":     home_runs,
        "away_score":     away_runs,
        "actual_total":   actual_total,
        "actual_winner":  pred_row["Home"] if home_won else pred_row["Away"],
        "model_pick":     pred_row["Home"] if model_picks_home else pred_row["Away"],
        "home_win_pct":   pred_row["Home_Win_Pct"],
        "away_win_pct":   pred_row["Away_Win_Pct"],
        "ml_correct":     ml_correct,
        "confidence":     max(pred_row["Home_Win_Pct"], pred_row["Away_Win_Pct"]),
        "xTotal_Runs":    xruns,
        "runs_error":     runs_error,
        "rl_cover":       rl_cover,
        "model_bias":     "HIGH" if xruns > actual_total else "LOW",
        "ou_line":        None,   # filled in after odds lookup
        "ou_result":      None,   # OVER/UNDER/PUSH vs Vegas line
    }


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(start, end, output_csv=False, game_filter=None, odds_file=None):

    print("\n" + "=" * 65)
    print("  BACKTEST: " + str(start) + " to " + str(end))
    print("=" * 65 + "\n")

    odds_data = {}
    if odds_file:
        odds_data = load_odds_file(odds_file)
        if odds_data:
            print("[backtest] historical odds loaded — will score value bets\n")
        else:
            print("[backtest] no odds data found — skipping bet scoring\n")

    all_results     = []
    all_bet_results = []
    current         = start

    while current <= end:
        if current >= date.today():
            current += timedelta(days=1)
            continue

        print("[" + str(current) + "] running model...", end=" ", flush=True)

        preds   = run_model_silent(current)
        actuals = fetch_actual_results(current)

        if preds.empty or actuals.empty:
            print("no data")
            current += timedelta(days=1)
            continue

        matched_pks = set()
        matched     = 0

        for _, pred in preds.iterrows():
            actual = None

            pk_match = actuals[actuals["game_pk"] == pred["game_pk"]]
            if not pk_match.empty:
                actual = pk_match.iloc[0]
            else:
                def nick(n):
                    parts = n.split()
                    if len(parts) >= 2 and parts[-1] in ("sox", "jays"):
                        return " ".join(parts[-2:])
                    return parts[-1] if parts else n

                ph = pred["home_team"]
                pa = pred["away_team"]
                for _, act in actuals.iterrows():
                    if act["game_pk"] in matched_pks:
                        continue
                    if nick(ph) == nick(act["home_team"]) and nick(pa) == nick(act["away_team"]):
                        actual = act
                        break

            if actual is None:
                continue

            matched_pks.add(actual["game_pk"])
            scored = score_game(pred, actual)
            all_results.append(scored)

            if odds_data:
                hist_odds = get_odds_for_game(
                    pred["home_team"], pred["away_team"],
                    current, odds_data
                )
                if hist_odds:
                    # Fill in ou_line and ou_result on the scored game
                    tot_line = hist_odds.get("tot_line")
                    if tot_line is not None:
                        scored["ou_line"] = tot_line
                        act_total = int(actual["home_runs"]) + int(actual["away_runs"])
                        if act_total > tot_line:
                            scored["ou_result"] = "OVER"
                        elif act_total < tot_line:
                            scored["ou_result"] = "UNDER"
                        else:
                            scored["ou_result"] = "PUSH"

                    bets = score_bet(
                        "all", hist_odds,
                        pred["Home_Win_Pct"] / 100,
                        pred["Away_Win_Pct"] / 100,
                        pred["xTotal_Runs"],
                        int(actual["home_runs"]),
                        int(actual["away_runs"]),
                        pred["Home"], pred["Away"],
                        min_edge=0.05
                    )
                    for b in bets:
                        b["date"] = str(current)
                        b["game"] = pred["Home"] + " vs " + pred["Away"]
                        all_bet_results.append(b)

            matched += 1

        print(str(matched) + " games matched")
        current += timedelta(days=1)

    if not all_results:
        print("\n[backtest] No results to report.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    if game_filter:
        mask = (df["home"].str.contains(game_filter, case=False) |
                df["away"].str.contains(game_filter, case=False))
        df = df[mask]
        if df.empty:
            print("\n[backtest] No games found matching '" + game_filter + "'")
            return df

    _print_backtest_report(df, start, end, game_filter)

    if all_bet_results:
        bets_df = pd.DataFrame(all_bet_results)
        _print_bet_report(bets_df)
        if output_csv:
            os.makedirs("results", exist_ok=True)
            bets_path = "results/backtest_bets_" + str(start) + "_" + str(end) + ".csv"
            bets_df.to_csv(bets_path, index=False)
            print("[backtest] bet results saved -> " + bets_path)

    if output_csv:
        os.makedirs("results", exist_ok=True)
        path = "results/backtest_" + str(start) + "_" + str(end) + ".csv"
        df.to_csv(path, index=False)
        print("[backtest] game results saved -> " + path)

    return df


# ── Summary report printer ────────────────────────────────────────────────────

def _print_backtest_report(df, start, end, game_filter=None):
    total   = len(df)
    ml_wins = int(df["ml_correct"].sum())
    ml_pct  = ml_wins / total * 100 if total > 0 else 0
    avg_err = df["runs_error"].mean()
    rl_wins = int(df["rl_cover"].sum())
    rl_pct  = rl_wins / total * 100 if total > 0 else 0

    high_conf = df[df["confidence"] >= 65]
    med_conf  = df[(df["confidence"] >= 55) & (df["confidence"] < 65)]
    low_conf  = df[df["confidence"] < 55]

    def bucket_str(sub):
        if sub.empty:
            return "0-0 (--)"
        w = int(sub["ml_correct"].sum())
        l = len(sub) - w
        p = w / len(sub) * 100
        return str(w) + "-" + str(l) + " (" + str(round(p)) + "%)"

    over_calls  = df[df["model_bias"] == "HIGH"]
    under_calls = df[df["model_bias"] == "LOW"]

    bar = "=" * 65
    title = "  BACKTEST RESULTS"
    if game_filter:
        title += " -- " + game_filter

    print("\n" + bar)
    print(title)
    print("  " + str(start) + " to " + str(end) + " | " + str(total) + " games")
    print(bar)

    print("\n  MONEYLINE ACCURACY")
    print("  Overall:                    " + str(ml_wins) + "-" + str(total - ml_wins) +
          " (" + str(round(ml_pct, 1)) + "%)")
    print("  High confidence (>=65%):    " + bucket_str(high_conf))
    print("  Medium confidence (55-65%): " + bucket_str(med_conf))
    print("  Low confidence (<55%):      " + bucket_str(low_conf))

    print("\n  RUN LINE (-1.5) ACCURACY")
    print("  Fav covers -1.5:            " + str(rl_wins) + "-" + str(total - rl_wins) +
          " (" + str(round(rl_pct, 1)) + "%)")

    print("\n  xRUNS CALIBRATION")
    print("  Avg error (runs):           " + str(round(avg_err, 2)))
    print("  Model HIGH (overestimated):  " + str(len(over_calls)) + " games")
    print("  Model LOW (underestimated): " + str(len(under_calls)) + " games")
    if total > 0:
        bias = (len(over_calls) - len(under_calls)) / total * 100
        direction = "HIGH" if bias > 0 else "LOW"
        print("  Model bias:                 " + str(round(abs(bias))) + "% " + direction)

    print("\n" + bar)
    print("  PER-GAME RESULTS")
    print(bar)

    GW = 32
    header = ("  " + "GAME".ljust(GW) + " " + "SCORE".rjust(7) + "  " +
              "PICK".rjust(5) + "  " + "ML".rjust(5) + "  " +
              "xRuns".rjust(6) + "  " + "ACT".rjust(4) + "  " +
              "ERR".rjust(4) + "  " + "RL".rjust(4) + "  " +
              "BIAS".rjust(5) + "  " + "O/U LINE".rjust(8))
    print(header)
    print("  " + "-" * GW + " " + "-" * 7 + "  " + "-" * 5 + "  " +
          "-" * 5 + "  " + "-" * 6 + "  " + "-" * 4 + "  " +
          "-" * 4 + "  " + "-" * 4 + "  " + "-" * 5 + "  " + "-" * 8)

    for dt, group in df.groupby("date"):
        print("\n  -- " + str(dt) + " --")
        for _, r in group.iterrows():
            game_str = (r["home"] + " vs " + r["away"])[:GW]
            score_str = str(int(r["home_score"])) + "-" + str(int(r["away_score"]))
            ml_icon  = "OK" if r["ml_correct"] else "XX"
            rl_icon  = "OK" if r["rl_cover"]   else "XX"
            conf_str = str(round(r["confidence"])) + "%"
            xruns    = str(round(r["xTotal_Runs"], 1))
            act      = str(int(r["actual_total"]))
            err      = str(round(r["runs_error"], 1))
            bias     = str(r.get("model_bias", ""))[:4]

            # O/U vs Vegas line
            ou_line   = r.get("ou_line")
            ou_result = r.get("ou_result", "")
            if ou_line is not None:
                ou_str = str(ou_line) + " " + str(ou_result)[:2]
            else:
                ou_str = "--"

            print("  " + game_str.ljust(GW) + " " + score_str.rjust(7) + "  " +
                  conf_str.rjust(5) + "  " + ml_icon.rjust(5) + "  " +
                  xruns.rjust(6) + "  " + act.rjust(4) + "  " +
                  err.rjust(4) + "  " + rl_icon.rjust(4) + "  " +
                  bias.rjust(5) + "  " + ou_str.rjust(8))

    print("\n" + bar)
    print("  ML OK/XX = moneyline correct/wrong | RL OK/XX = fav covered -1.5")
    print("  BIAS HIGH/LOW = model over/under estimated runs vs actual")
    print("  O/U LINE = Vegas total | OV=went over UN=went under PU=push")
    print(bar + "\n")


# ── Bet report printer ────────────────────────────────────────────────────────

def _print_bet_report(bets_df):
    total_bets   = len(bets_df)
    if total_bets == 0:
        return

    wins         = int(bets_df["won"].sum())
    losses       = total_bets - wins
    hit_rate     = round(wins / total_bets * 100, 1)
    total_profit = round(bets_df["profit"].sum(), 2)
    roi          = round(total_profit / total_bets * 100, 1)
    bar          = "=" * 65

    print("\n" + bar)
    print("  HISTORICAL BET SCORING")
    print(bar)
    print("  Total flagged bets : " + str(total_bets))
    print("  Record             : " + str(wins) + "-" + str(losses) +
          " (" + str(hit_rate) + "%)")
    print("  Net profit ($1/bet): $" + str(total_profit))
    print("  ROI                : " + str(roi) + "%")

    print("\n  By type:")
    for bt in ["Moneyline", "Run Line", "Total"]:
        sub = bets_df[bets_df["type"] == bt]
        if sub.empty:
            continue
        sw  = int(sub["won"].sum())
        sl  = len(sub) - sw
        sp  = round(sub["profit"].sum(), 2)
        pct = round(sw / len(sub) * 100)
        print("  " + bt.ljust(12) + str(sw) + "-" + str(sl) +
              " (" + str(pct) + "%)  profit: $" + str(sp))

    print("\n  By edge tier:")
    tiers = [
        ("Strong (>=12%)", 0.12, 1.0),
        ("Good   (8-12%)", 0.08, 0.12),
        ("Lean    (5-8%)", 0.05, 0.08),
    ]
    for label, lo, hi in tiers:
        sub = bets_df[(bets_df["edge"] >= lo) & (bets_df["edge"] < hi)]
        if sub.empty:
            continue
        sw = int(sub["won"].sum())
        sl = len(sub) - sw
        sp = round(sub["profit"].sum(), 2)
        print("  " + label.ljust(18) + str(sw) + "-" + str(sl) +
              "  profit: $" + str(sp))

    print("\n  Top 5 bets by edge:")
    top = bets_df.nlargest(5, "edge")
    for _, r in top.iterrows():
        result = "WIN " if r["won"] else "LOSS"
        game   = str(r["game"])[:35].ljust(35)
        bet    = str(r["bet"]).ljust(22)
        edge   = str(round(r["edge"] * 100, 1))
        profit = str(round(r["profit"], 2))
        print("  " + game + " " + bet + " edge " + edge +
              "%  " + result + "  $" + profit)

    print("\n" + bar + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Model Backtest")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, required=True,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--csv",   action="store_true",
                        help="Save results to CSV")
    parser.add_argument("--game",  type=str, default=None,
                        help="Filter to games containing this team name")
    parser.add_argument("--odds",  type=str, default=None,
                        help="Path to scraped odds JSON for bet scoring")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date   = date.fromisoformat(args.end)

    run_backtest(start_date, end_date,
                 output_csv=args.csv,
                 game_filter=args.game,
                 odds_file=args.odds)
