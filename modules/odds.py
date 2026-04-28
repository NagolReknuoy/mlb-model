# =============================================================================
# odds.py — Fetch live sportsbook odds and identify value bets
#
# Uses The Odds API (the-odds-api.com) free tier
# Set your API key in the ODDS_API_KEY variable below
# =============================================================================

import requests
import pandas as pd
import numpy as np
from datetime import date
from scipy.stats import poisson
from modules.utils import clean_name

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Key is read from environment variable ODDS_API_KEY (set in GitHub Secrets)
# For local use, set it in your terminal:
#   Windows: set ODDS_API_KEY=your_key_here
#   Mac/Linux: export ODDS_API_KEY=your_key_here
# OR paste directly below for local testing only (never commit to GitHub)
import os as _os
ODDS_API_KEY  = _os.environ.get("ODDS_API_KEY", "YOUR_API_KEY_HERE")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
REGIONS       = "us"          # us odds (American format)
MARKETS       = "h2h,spreads,totals"
ODDS_FORMAT   = "american"
MIN_EDGE      = 0.07          # minimum edge to flag a bet (7%)


# ── American odds → implied probability (with vig removed) ───────────────────

def american_to_prob(odds: float) -> float:
    """Convert American moneyline odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def remove_vig(home_prob: float, away_prob: float) -> tuple:
    """Remove bookmaker vig so probabilities sum to 100%."""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


# ── Fetch odds from The Odds API ─────────────────────────────────────────────

def fetch_odds(target_date: date = None) -> pd.DataFrame:
    """
    Fetch MLB moneyline, run line, and totals for today's games.
    Returns a DataFrame with one row per game.
    """
    if ODDS_API_KEY == "YOUR_API_KEY_HERE":
        print("[odds] ⚠️  No API key set — skipping odds fetch")
        return pd.DataFrame()

    from datetime import datetime, timezone
    # Only pull games that haven't started yet — excludes live games
    # where odds shift dramatically (e.g. team down 8 runs = +3300)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = f"{ODDS_API_BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey":             ODDS_API_KEY,
        "regions":            REGIONS,
        "markets":            MARKETS,
        "oddsFormat":         ODDS_FORMAT,
        "dateFormat":         "iso",
        "commenceTimeFrom":   now_utc,   # only future/upcoming games
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"[odds] API credits used: {used} | remaining: {remaining}")
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[odds] API error: {e}")
        return pd.DataFrame()

    if not data:
        print("[odds] no games found in API response")
        return pd.DataFrame()

    rows = []
    for game in data:
        home = clean_name(game.get("home_team", ""))
        away = clean_name(game.get("away_team", ""))
        game_id = game.get("id", "")

        # Aggregate odds across all bookmakers (use median to reduce outliers)
        ml_home, ml_away = [], []
        rl_home, rl_away = [], []
        rl_line_vals     = []
        tot_over, tot_under = [], []
        tot_line_vals    = []

        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                key = market.get("key")
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if key == "h2h":
                    h = outcomes.get(game.get("home_team", ""), {}).get("price")
                    a = outcomes.get(game.get("away_team", ""), {}).get("price")
                    if h and a:
                        ml_home.append(h)
                        ml_away.append(a)

                elif key == "spreads":
                    h = outcomes.get(game.get("home_team", ""), {})
                    a = outcomes.get(game.get("away_team", ""), {})
                    if h.get("price") and a.get("price"):
                        rl_home.append(h["price"])
                        rl_away.append(a["price"])
                        rl_line_vals.append(h.get("point", -1.5))

                elif key == "totals":
                    ov = outcomes.get("Over",  {})
                    un = outcomes.get("Under", {})
                    if ov.get("price") and un.get("price"):
                        tot_over.append(ov["price"])
                        tot_under.append(un["price"])
                        tot_line_vals.append(ov.get("point", 8.5))

        def med(lst):
            return float(np.median(lst)) if lst else None

        rows.append({
            "home_team":   home,
            "away_team":   away,
            "game_id":     game_id,
            # Moneyline
            "ml_home":     med(ml_home),
            "ml_away":     med(ml_away),
            # Run line
            "rl_home":     med(rl_home),
            "rl_away":     med(rl_away),
            "rl_line":     med(rl_line_vals) or -1.5,
            # Totals
            "tot_over":    med(tot_over),
            "tot_under":   med(tot_under),
            "tot_line":    med(tot_line_vals) or 8.5,
        })

    print(f"[odds] fetched odds for {len(rows)} games")
    return pd.DataFrame(rows)


# ── Over/under probability from Poisson ──────────────────────────────────────

def poisson_over_prob(expected_total: float, line: float) -> float:
    """
    Probability that combined runs exceed the line using Poisson distribution.
    Splits expected total evenly between teams.
    """
    lam_each = expected_total / 2
    # P(total > line) = 1 - P(total <= floor(line))
    threshold = int(np.floor(line))
    prob_over = 0.0
    for h in range(0, 20):
        for a in range(0, 20):
            if h + a > threshold:
                prob_over += (poisson.pmf(h, lam_each) * 
                              poisson.pmf(a, lam_each))
    return prob_over


# ── Run line cover probability ────────────────────────────────────────────────

def poisson_cover_prob(lambda_home: float, lambda_away: float, 
                       spread: float) -> float:
    """
    Probability home team covers the spread (default -1.5).
    P(home wins by more than abs(spread))
    """
    prob = 0.0
    margin_needed = abs(spread)
    for h in range(0, 20):
        for a in range(0, 20):
            if h - a > margin_needed:
                prob += (poisson.pmf(h, max(0.1, lambda_home)) *
                         poisson.pmf(a, max(0.1, lambda_away)))
    return prob


# ── Main value bet finder ─────────────────────────────────────────────────────

def find_value_bets(predictions: pd.DataFrame, 
                    odds_df: pd.DataFrame,
                    min_edge: float = MIN_EDGE) -> pd.DataFrame:
    """
    Compare model probabilities to sportsbook implied probabilities.
    Returns DataFrame of value bets sorted by edge descending.
    """
    if odds_df.empty or predictions.empty:
        print("[odds] no odds data to compare")
        return pd.DataFrame()

    bets = []

    for _, pred in predictions.iterrows():
        home = clean_name(pred["Home"])
        away = clean_name(pred["Away"])

        # Match to odds — require BOTH teams to match
        odds_row = None
        for _, o in odds_df.iterrows():
            oh = clean_name(o["home_team"])
            oa = clean_name(o["away_team"])

            # Extract last word of team name as nickname for matching
            # e.g. "los angeles angels" -> "angels"
            def nickname(n):
                parts = n.split()
                if len(parts) >= 2 and parts[-1] in ("sox", "jays"):
                    return " ".join(parts[-2:])
                return parts[-1] if parts else n

            nh, na = nickname(home), nickname(away)
            onh, ona = nickname(oh), nickname(oa)

            # Both home and away must match
            home_match = (nh == onh or nh in oh or oh in home)
            away_match = (na == ona or na in oa or oa in away)

            if home_match and away_match:
                odds_row = o
                break

        if odds_row is None:
            continue

        model_home_pct = pred["Home_Win_Pct"] / 100
        model_away_pct = pred["Away_Win_Pct"] / 100
        model_xruns    = pred["xTotal_Runs"]
        # Clamp lambdas to same range used in simulation
        # Raw lambdas can be extreme; clamping prevents unrealistic cover probs
        lh = max(3.5, min(5.5, pred.get("Home_lambda", model_xruns / 2)))
        la = max(3.5, min(5.5, pred.get("Away_lambda", model_xruns / 2)))

        # ── Moneyline ────────────────────────────────────────────────────────
        # Sanity check: MLB moneylines should be between -500 and +500
        ml_h_ok = (odds_row["ml_home"] and -500 <= odds_row["ml_home"] <= 500)
        ml_a_ok = (odds_row["ml_away"] and -500 <= odds_row["ml_away"] <= 500)
        if ml_h_ok and ml_a_ok:
            raw_h = american_to_prob(odds_row["ml_home"])
            raw_a = american_to_prob(odds_row["ml_away"])
            book_h, book_a = remove_vig(raw_h, raw_a)

            edge_h = model_home_pct - book_h
            edge_a = model_away_pct - book_a

            if abs(edge_h) >= min_edge:
                side = pred["Home"] if edge_h > 0 else pred["Away"]
                ml   = odds_row["ml_home"] if edge_h > 0 else odds_row["ml_away"]
                bets.append({
                    "Game":       f"{pred['Home']} vs {pred['Away']}",
                    "Bet":        f"{side} ML",
                    "Odds":       f"{int(ml):+d}",
                    "Book_Prob":  f"{(book_h if edge_h > 0 else book_a)*100:.1f}%",
                    "Model_Prob": f"{(model_home_pct if edge_h > 0 else model_away_pct)*100:.1f}%",
                    "Edge":       abs(edge_h),
                    "Rating":     _rate(abs(edge_h)),
                    "Type":       "Moneyline",
                })

        # ── Run line ─────────────────────────────────────────────────────────
        rl_h_ok = (odds_row["rl_home"] and -400 <= odds_row["rl_home"] <= 400)
        rl_a_ok = (odds_row["rl_away"] and -400 <= odds_row["rl_away"] <= 400)
        if rl_h_ok and rl_a_ok:
            rl = abs(odds_row["rl_line"])
            model_cover_h = poisson_cover_prob(lh, la,  rl)
            model_cover_a = poisson_cover_prob(la, lh,  rl)
            book_rl_h = american_to_prob(odds_row["rl_home"])
            book_rl_a = american_to_prob(odds_row["rl_away"])
            book_rl_h, book_rl_a = remove_vig(book_rl_h, book_rl_a)

            edge_rl_h = model_cover_h - book_rl_h
            edge_rl_a = model_cover_a - book_rl_a

            for edge, side, ml, book_p, model_p in [
                (edge_rl_h, f"{pred['Home']} -{rl}", odds_row["rl_home"], book_rl_h, model_cover_h),
                (edge_rl_a, f"{pred['Away']} +{rl}", odds_row["rl_away"], book_rl_a, model_cover_a),
            ]:
                if min_edge <= edge <= 0.20:  # cap at 20% — higher = likely error
                    bets.append({
                        "Game":       f"{pred['Home']} vs {pred['Away']}",
                        "Bet":        side,
                        "Odds":       f"{int(ml):+d}",
                        "Book_Prob":  f"{book_p*100:.1f}%",
                        "Model_Prob": f"{model_p*100:.1f}%",
                        "Edge":       edge,
                        "Rating":     _rate(edge),
                        "Type":       "Run Line",
                    })

        # ── Totals (over/under) ───────────────────────────────────────────────
        tot_line = odds_row["tot_line"]
        if odds_row["tot_over"] and odds_row["tot_under"] and tot_line:
            line = tot_line
            model_over  = poisson_over_prob(model_xruns, line)
            model_under = 1 - model_over
            book_over   = american_to_prob(odds_row["tot_over"])
            book_under  = american_to_prob(odds_row["tot_under"])
            book_over, book_under = remove_vig(book_over, book_under)

            edge_over  = model_over  - book_over
            edge_under = model_under - book_under

            for edge, label, ml, book_p, model_p in [
                (edge_over,  f"OVER  {line}",  odds_row["tot_over"],  book_over,  model_over),
                (edge_under, f"UNDER {line}", odds_row["tot_under"], book_under, model_under),
            ]:
                if min_edge <= edge <= 0.25:  # cap at 25% — higher = likely mismatch
                    bets.append({
                        "Game":       f"{pred['Home']} vs {pred['Away']}",
                        "Bet":        label,
                        "Odds":       f"{int(ml):+d}",
                        "Book_Prob":  f"{book_p*100:.1f}%",
                        "Model_Prob": f"{model_p*100:.1f}%",
                        "Edge":       edge,
                        "Rating":     _rate(edge),
                        "Type":       "Total",
                    })

    if not bets:
        print("[odds] no value bets found above edge threshold")
        return pd.DataFrame()

    df = pd.DataFrame(bets).sort_values("Edge", ascending=False).reset_index(drop=True)
    return df


def _rate(edge: float) -> str:
    if edge >= 0.12:
        return "Strong"
    elif edge >= 0.08:
        return "Good"
    elif edge >= 0.05:
        return "Lean"
    return ""


def print_value_bets(bets_df: pd.DataFrame):
    if bets_df.empty:
        print("\n[odds] No value bets today above the edge threshold.")
        return

    W_GAME   = 36
    W_BET    = 26
    W_ODDS   = 7
    W_BOOK   = 7
    W_MODEL  = 8
    W_EDGE   = 7
    W_RATING = 16
    total = W_GAME + W_BET + W_ODDS + W_BOOK + W_MODEL + W_EDGE + W_RATING + 8
    bar   = "=" * total

    print(f"\n{bar}")
    print(f"  💰 VALUE BETS — {len(bets_df)} found")
    print(f"{bar}")
    print(f"  {'GAME':<{W_GAME}} {'BET':<{W_BET}} {'ODDS':>{W_ODDS}} "
          f"{'BOOK%':>{W_BOOK}} {'MODEL%':>{W_MODEL}} {'EDGE':>{W_EDGE}}  RATING")
    print(f"{'-' * total}")

    for bet_type in ["Moneyline", "Run Line", "Total"]:
        subset = bets_df[bets_df["Type"] == bet_type]
        if subset.empty:
            continue

        # Section header
        print(f"  ── {bet_type} {'─' * (total - len(bet_type) - 6)}")

        for _, row in subset.iterrows():
            game = row['Game']
            if len(game) > W_GAME:
                game = game[:W_GAME-1]
            bet = row['Bet']
            if len(bet) > W_BET:
                bet = bet[:W_BET-1]

            print(f"  {game:<{W_GAME}} {bet:<{W_BET}} {row['Odds']:>{W_ODDS}} "
                  f"{row['Book_Prob']:>{W_BOOK}} {row['Model_Prob']:>{W_MODEL}} "
                  f"{row['Edge']*100:>{W_EDGE-1}.1f}%  {row['Rating']}")

        print(f"{'-' * total}")

    print(f"{bar}")
    print(f"  ⚠️  For entertainment only. Please gamble responsibly.")
    print(f"{bar}\n")
