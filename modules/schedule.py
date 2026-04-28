# =============================================================================
# schedule.py — MLB schedule + team offensive/defensive strength
# Park-neutralized OS/DS so Coors doesn't inflate Rockies ratings
# =============================================================================

import requests
import pandas as pd
from datetime import date, timedelta
from modules.utils import clean_name

# Import park factors for neutralization
# These are the same annual factors from parks.py
PARK_FACTORS = {
    "los angeles angels":      0.99,
    "houston astros":          0.97,
    "oakland athletics":       0.97,
    "toronto blue jays":       1.01,
    "atlanta braves":          1.02,
    "milwaukee brewers":       0.96,
    "st. louis cardinals":     0.99,
    "chicago cubs":            1.04,
    "arizona diamondbacks":    1.05,
    "los angeles dodgers":     0.96,
    "san francisco giants":    0.94,
    "cleveland guardians":     0.98,
    "seattle mariners":        0.97,
    "miami marlins":           0.99,
    "new york mets":           0.97,
    "washington nationals":    1.00,
    "baltimore orioles":       1.02,
    "san diego padres":        0.95,
    "philadelphia phillies":   1.04,
    "pittsburgh pirates":      0.96,
    "texas rangers":           1.03,
    "tampa bay rays":          0.98,
    "boston red sox":          1.03,
    "cincinnati reds":         1.06,
    "colorado rockies":        1.13,
    "kansas city royals":      1.00,
    "detroit tigers":          0.97,
    "minnesota twins":         1.00,
    "chicago white sox":       1.01,
    "new york yankees":        1.03,
}


def _get_park_factor(team: str) -> float:
    """Return park factor for a team, with fuzzy matching."""
    t = clean_name(team)
    if t in PARK_FACTORS:
        return PARK_FACTORS[t]
    for key in PARK_FACTORS:
        if key in t or t in key:
            return PARK_FACTORS[key]
    return 1.0


def load_season_data(today: date = None, rolling_days: int = 30) -> dict:
    if today is None:
        today = date.today()

    season = today.year
    print(f"[schedule] fetching MLB schedule for {season} ...")

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&season={season}&gameType=R"
        f"&hydrate=team,linescore"
        f"&startDate={season}-01-01&endDate={season}-12-31"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            teams  = game.get("teams", {})
            rows.append({
                "game_pk":    game["gamePk"],
                "date":       date_entry["date"],
                "home_team":  clean_name(teams.get("home", {}).get("team", {}).get("name", "")),
                "away_team":  clean_name(teams.get("away", {}).get("team", {}).get("name", "")),
                "home_runs":  teams.get("home", {}).get("score"),
                "away_runs":  teams.get("away", {}).get("score"),
                "status":     status,
                "game_number": game.get("gameNumber", 1),  # 1 or 2 for doubleheaders
            })

    games = pd.DataFrame(rows)
    games["date"] = pd.to_datetime(games["date"]).dt.date

    past = games[games["status"].str.lower() == "final"].dropna(
        subset=["home_runs", "away_runs"]
    ).copy()
    past["home_runs"] = past["home_runs"].astype(float)
    past["away_runs"] = past["away_runs"].astype(float)

    if past.empty:
        raise RuntimeError(f"[schedule] No completed games found for {season}")

    # ── Park-neutralize runs before computing team strength ───────────────────
    # Divide each game's runs by the home park factor so Coors-inflated
    # Rockies scores don't make them look like a great offensive team on the road.
    past["park_factor"] = past["home_team"].apply(_get_park_factor)
    past["home_runs_neutral"] = past["home_runs"] / past["park_factor"]
    past["away_runs_neutral"] = past["away_runs"] / past["park_factor"]

    league_avg = (past["home_runs"].mean() + past["away_runs"].mean()) / 2
    home_field_mult = past["home_runs"].mean() / league_avg
    away_field_mult = past["away_runs"].mean() / league_avg

    print(f"[schedule] {len(past)} completed games | league avg {league_avg:.2f} R/team/game")

    # Rolling window
    cutoff = today - timedelta(days=rolling_days)
    recent = past[past["date"] >= cutoff]
    if len(recent) < 50:
        print("[schedule] rolling window thin – using full season")
        recent = past

    # ── Team strength using park-neutralized runs ─────────────────────────────
    home_df = recent[["home_team","home_runs_neutral","away_runs_neutral"]].rename(
        columns={"home_team":"team","home_runs_neutral":"rs","away_runs_neutral":"ra"})
    away_df = recent[["away_team","away_runs_neutral","home_runs_neutral"]].rename(
        columns={"away_team":"team","away_runs_neutral":"rs","home_runs_neutral":"ra"})
    both = pd.concat([home_df, away_df], ignore_index=True)

    team_stats = both.groupby("team").agg(
        G=("rs","count"), RS=("rs","sum"), RA=("ra","sum")
    ).reset_index()
    team_stats["OS"] = ((team_stats["RS"] / team_stats["G"]) / league_avg).clip(0.75, 1.35)
    team_stats["DS"] = ((team_stats["RA"] / team_stats["G"]) / league_avg).clip(0.75, 1.35)

    # Today's games
    today_games = games[games["date"] == today][
        ["game_pk","home_team","away_team","game_number"]
    ].drop_duplicates(subset=["game_pk"]).reset_index(drop=True)

    print(f"[schedule] {len(today_games)} games today")

    return {
        "games":           games,
        "past_games":      past,
        "today_games":     today_games,
        "league_avg_runs": league_avg,
        "team_stats":      team_stats,
        "home_field_mult": home_field_mult,
        "away_field_mult": away_field_mult,
    }
