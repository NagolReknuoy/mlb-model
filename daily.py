# =============================================================================
# daily.py — Run the full MLB model pipeline automatically
#
# Usage:
#   python daily.py                  # run everything for today
#   python daily.py --no-odds        # skip live odds API (saves credits)
#   python daily.py --date 2026-04-26  # run for a specific date
#
# What it does in order:
#   1. Scrapes yesterday's historical odds and merges into mlb_odds_2026.json
#   2. Scores yesterday's bets vs actual results
#   3. Runs today's predictions + live value bets
#   4. Prints a summary of everything
# =============================================================================

import argparse
import subprocess
import sys
import os
from datetime import date, timedelta


ODDS_FILE = "mlb_odds_2026.json"


def run(cmd: list, label: str) -> int:
    """Run a command and stream output. Returns exit code."""
    print("\n" + "=" * 65)
    print("  " + label)
    print("=" * 65 + "\n")
    result = subprocess.run([sys.executable] + cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="MLB Model Daily Pipeline")
    parser.add_argument("--date",     type=str, default=None,
                        help="Today's date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-odds",  action="store_true",
                        help="Skip live odds API fetch (saves credits)")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip scraping yesterday's historical odds")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Skip yesterday's bet scoring backtest")
    args = parser.parse_args()

    today     = date.fromisoformat(args.date) if args.date else date.today()
    yesterday = today - timedelta(days=1)

    print("\n" + "=" * 65)
    print("  MLB MODEL DAILY PIPELINE")
    print("  Today:     " + str(today))
    print("  Yesterday: " + str(yesterday))
    print("=" * 65)

    # ── Step 1: Scrape yesterday's odds ──────────────────────────────────────
    if not args.no_scrape:
        yest_str = str(yesterday)
        odds_exists = os.path.exists(ODDS_FILE)
        scrape_cmd = [
            "scraper.py",
            yest_str, yest_str,
            "-t", "moneyline", "pointspread", "totals",
            "-o", ODDS_FILE,
        ]
        if odds_exists:
            scrape_cmd.append("-m")   # merge if file already exists
            label = "STEP 1: Scraping " + yest_str + " odds (merging into " + ODDS_FILE + ")"
        else:
            label = "STEP 1: Scraping " + yest_str + " odds (creating " + ODDS_FILE + ")"

        code = run(scrape_cmd, label)
        if code != 0:
            print("[daily] WARNING: scraper failed — continuing anyway")
    else:
        print("\n[daily] Skipping odds scrape (--no-scrape)")

    # ── Step 2: Backtest yesterday ────────────────────────────────────────────
    if not args.no_backtest:
        bt_cmd = [
            "backtest.py",
            "--start", str(yesterday),
            "--end",   str(yesterday),
            "--csv",
        ]
        if os.path.exists(ODDS_FILE):
            bt_cmd += ["--odds", ODDS_FILE]

        code = run(bt_cmd, "STEP 2: Scoring yesterday's bets (" + str(yesterday) + ")")
        if code != 0:
            print("[daily] WARNING: backtest failed — continuing anyway")
    else:
        print("\n[daily] Skipping backtest (--no-backtest)")

    # ── Step 3: Today's predictions + value bets ──────────────────────────────
    pred_cmd = [
        "run_model.py",
        "--date", str(today),
        "--csv",
    ]
    if args.no_odds:
        pred_cmd.append("--no-odds")

    code = run(pred_cmd, "STEP 3: Today's predictions (" + str(today) + ")")
    if code != 0:
        print("[daily] ERROR: predictions failed")
        sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DAILY PIPELINE COMPLETE")
    print("  Predictions saved → results/predictions_" + str(today) + ".csv")
    print("  Value bets saved  → results/value_bets_" + str(today) + ".csv")
    if not args.no_backtest:
        print("  Yesterday bets    → results/backtest_bets_" + str(yesterday) + "_" + str(yesterday) + ".csv")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
