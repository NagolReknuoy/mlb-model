import argparse
import os
import aiohttp
import asyncio
import sys

# Windows requires SelectorEventLoop for aiodns/aiohttp to work
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import json
import re
import random
import time
import functools
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
from pools import USER_AGENTS, ACCEPT_LANGUAGES

NEXT_DATA_PATTERN = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL)

@functools.lru_cache(maxsize=64)
def normalize_name(name):
    """Cached team name normalization"""
    return (name
            .lower()
            .replace(".", "")
            .replace("'", "")
            .replace("-", " ")
            .replace("&", "and")
            .strip())

async def get_html_async(session, url, semaphore, retries=3, base_delay=2):
    """Async version of get_html with semaphore for rate limiting"""
    for attempt in range(retries):
        async with semaphore:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": random.choice(ACCEPT_LANGUAGES),
            }
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    else:
                        print(f"Request failed with status {resp.status} for {url}")
            except Exception as e:
                print(f"Error fetching {url}: {e}")

        if attempt < retries - 1:
            delay = base_delay + random.uniform(0, 2)
            await asyncio.sleep(delay)

    return None

def get_mlb_schedule(start_date, end_date):
    """Fetch MLB schedule to get game types (Regular Season, Playoffs, etc.)"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    schedule_map = {}
    current_start = start

    while current_start <= end:
        # chunk by year to avoid overly large requests
        current_end = min(
            datetime(current_start.year, 12, 31),
            end
        )
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1"
            f"&startDate={current_start.strftime('%Y-%m-%d')}"
            f"&endDate={current_end.strftime('%Y-%m-%d')}"
        )
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            for date_info in data.get("dates", []):
                date = date_info["date"]
                if date not in schedule_map:
                    schedule_map[date] = {}
                for g in date_info.get("games", []):
                    away = normalize_name(g["teams"]["away"]["team"]["name"])
                    home = normalize_name(g["teams"]["home"]["team"]["name"])
                    schedule_map[date][(away, home)] = g.get("gameType", "R")
        except Exception as e:
            print(f"Error fetching schedule for {current_start} - {current_end}: {e}")

        current_start = current_end + timedelta(days=1)

    return schedule_map

def get_odds_url(date, odds_type):
    """Get the appropriate SportsBookReview URL for the given odds type"""
    base_url = "https://www.sportsbookreview.com/betting-odds/mlb-baseball"
    if odds_type == "moneyline":
        return f"{base_url}/?date={date}"
    elif odds_type == "pointspread":
        return f"{base_url}/pointspread/full-game/?date={date}"
    elif odds_type == "totals":
        return f"{base_url}/totals/full-game/?date={date}"
    else:
        raise ValueError(f"Unknown odds type: {odds_type}")

def extract_odds_data(odds, odds_type):
    """Extract the appropriate odds data based on the odds type"""
    opening_line = odds.get("openingLine", {})
    current_line = odds.get("currentLine", {})

    if odds_type == "moneyline":
        keys = ["homeOdds", "awayOdds"]
    elif odds_type == "pointspread":
        keys = ["homeOdds", "awayOdds", "homeSpread", "awaySpread"]
    else:  # totals
        keys = ["overOdds", "underOdds", "total"]

    return (
        {k: opening_line.get(k) for k in keys},
        {k: current_line.get(k) for k in keys},
    )

async def scrape_mlb_odds_async(session, date, odds_type, game_type_map, semaphore, base_delay=2):
    """Scrape odds for one date + odds type"""
    url  = get_odds_url(date, odds_type)
    html = await get_html_async(session, url, semaphore, base_delay=base_delay)

    if not html:
        print(f"Failed to fetch {odds_type} odds for {date}")
        return date, odds_type, []

    match = NEXT_DATA_PATTERN.search(html)
    if not match:
        print(f"No __NEXT_DATA__ found for {odds_type} odds on {date}")
        return date, odds_type, []

    try:
        data       = json.loads(match.group(1))
        odds_tables = data.get("props", {}).get("pageProps", {}).get("oddsTables", [])
        if not odds_tables:
            return date, odds_type, []
        game_rows = odds_tables[0].get("oddsTableModel", {}).get("gameRows", [])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing {odds_type} data for {date}: {e}")
        return date, odds_type, []

    games_for_date = []
    for game in game_rows:
        try:
            game_view = game.get("gameView", {})
            away = normalize_name(game_view.get("awayTeam", {}).get("fullName", "Unknown"))
            home = normalize_name(game_view.get("homeTeam", {}).get("fullName", "Unknown"))
            game_key = f"{away}_vs_{home}"

            cleaned_game = {
                "gameKey": game_key,
                "gameView": {
                    k: game_view.get(k)
                    for k in ["startDate", "awayTeam", "awayTeamScore",
                              "homeTeam", "homeTeamScore", "gameStatusText", "venueName"]
                },
            }
            cleaned_game["gameView"]["gameType"] = (
                game_type_map.get(date, {}).get((away, home), "Unknown")
            )

            cleaned_odds = []
            for odds in game.get("oddsViews", []):
                if odds is None:
                    continue
                try:
                    opening, current = extract_odds_data(odds, odds_type)
                    cleaned_odds.append({
                        "sportsbook":  odds.get("sportsbook", "Unknown"),
                        "openingLine": opening,
                        "currentLine": current,
                    })
                except Exception as e:
                    print(f"Error processing {odds_type} odds for {date}: {e}")

            cleaned_game["oddsViews"] = cleaned_odds
            games_for_date.append(cleaned_game)

        except Exception as e:
            print(f"Error processing game for {odds_type} on {date}: {e}")

    return date, odds_type, games_for_date

def merge_odds_data(all_results, odds_types):
    """Merge moneyline / pointspread / totals into one structure per game"""
    date_results = {}
    for date, odds_type, games in all_results:
        date_results.setdefault(date, {})[odds_type] = games

    merged_data = {}
    for date, odds_by_type in date_results.items():
        merged_games = {}
        for odds_type, games in odds_by_type.items():
            for game in games:
                gk = game.get("gameKey")
                if not gk:
                    continue
                if gk not in merged_games:
                    merged_games[gk] = {
                        "gameView": game["gameView"].copy(),
                        "odds": {}
                    }
                merged_games[gk]["odds"][odds_type] = game["oddsViews"]
        merged_data[date] = list(merged_games.values())

    return merged_data

async def scrape_range_async(start_date, end_date, fast, max_concurrent, odds_types):
    """Main async scraping loop"""
    print("Fetching MLB schedule...")
    game_type_map = get_mlb_schedule(start_date, end_date)

    if not game_type_map:
        print("No games found in date range")
        return {}

    dates      = sorted(game_type_map.keys())
    semaphore  = asyncio.Semaphore(max_concurrent)
    base_delay = 0.1 if fast else 1.0

    print(f"Found {len(dates)} dates | odds types: {', '.join(odds_types)}")

    async with aiohttp.ClientSession() as session:
        tasks = [
            scrape_mlb_odds_async(session, date, ot, game_type_map, semaphore, base_delay)
            for date in dates
            for ot in odds_types
        ]

        print(f"Scraping {len(tasks)} requests ({max_concurrent} concurrent)...")
        results    = []
        chunk_size = max_concurrent * 2
        pbar       = tqdm(total=len(tasks), desc="Scraping", unit="req")

        try:
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i:i + chunk_size]
                chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
                for r in chunk_results:
                    if isinstance(r, tuple) and len(r) == 3:
                        results.append(r)
                    else:
                        print(f"\nTask failed: {r}")
                pbar.update(len(chunk))
                if not fast and i + chunk_size < len(tasks):
                    await asyncio.sleep(base_delay + random.uniform(0, base_delay))
        finally:
            pbar.close()

    return merge_odds_data(results, odds_types)

def main():
    parser = argparse.ArgumentParser(description="MLB historical odds scraper (SportsBookReview)")
    parser.add_argument("start_date", help="Start date YYYY-MM-DD")
    parser.add_argument("end_date",   help="End date YYYY-MM-DD")
    parser.add_argument("-f", "--fast",       action="store_true",
                        help="Faster scraping (reduced delays — use carefully)")
    parser.add_argument("-c", "--concurrent", type=int, default=5,
                        help="Max concurrent requests (default 5, max 20)")
    parser.add_argument("-o", "--output",     default="mlb_odds.json",
                        help="Output JSON filename")
    parser.add_argument("-t", "--types",      nargs="+",
                        default=["moneyline"],
                        choices=["moneyline", "pointspread", "totals"],
                        help="Odds types to scrape (default: moneyline)")
    parser.add_argument("-m", "--merge",      action="store_true",
                        help="Merge new dates into existing output file instead of overwriting")
    args = parser.parse_args()

    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date,   "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD.")
        return

    if not (1 <= args.concurrent <= 20):
        print("Concurrent requests must be between 1 and 20")
        return

    odds_types = list(set(args.types))
    print(f"Scraping {', '.join(odds_types)} odds from {args.start_date} to {args.end_date}")
    start_time = time.time()

    all_data = asyncio.run(scrape_range_async(
        args.start_date, args.end_date,
        args.fast, args.concurrent, odds_types
    ))

    # Merge with existing file if requested
    if args.merge and os.path.exists(args.output):
        print(f"Merging with existing {args.output} ...")
        with open(args.output, "r") as f:
            existing = json.load(f)
        new_dates = 0
        for date_key, games in all_data.items():
            if date_key not in existing:
                existing[date_key] = games
                new_dates += 1
            else:
                print(f"  [merge] {date_key} already exists — skipping")
        all_data = existing
        print(f"  Added {new_dates} new dates")

    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=2)

    total_games = sum(len(g) for g in all_data.values())
    print(f"\nDone! {total_games} games from {len(all_data)} dates")
    print(f"Runtime: {time.time() - start_time:.1f}s")
    print(f"Saved → {args.output}")

if __name__ == "__main__":
    main()
