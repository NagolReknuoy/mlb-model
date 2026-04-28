# =============================================================================
# parks.py — Park factors + ballpark dimensions + monthly adjustments
# =============================================================================

import pandas as pd
import numpy as np
from datetime import date
from modules.utils import clean_name


# ── Annual park factors (FanGraphs 2025, 100 = neutral) ──────────────────────
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

# ── Monthly deviations (percentage points from annual, FanGraphs heatmap 2025)
# Apr, May, Jun, Jul, Aug, Sep
MONTHLY_DEV = {
    "los angeles angels":      [ 4,  1, -2,  5, -2, -6],
    "baltimore orioles":       [-6, -2,  4,  3,  6, -4],
    "boston red sox":          [-4, -1, -1,  3,  0,  2],
    "chicago white sox":       [ 1,  0, -2, 12, -5, -6],
    "cleveland guardians":     [-2, -4,  6, -1,  3, -2],
    "kansas city royals":      [-1,  1, -4, -4,  6,  2],
    "oakland athletics":       [-4,  5, -5,  3, -6,  6],
    "tampa bay rays":          [-1,  2,  2,  0,  0, -2],
    "toronto blue jays":       [ 0, -6,  6, -3,  3, -1],
    "arizona diamondbacks":    [-4, -3,  2, -1,  7, -2],
    "chicago cubs":            [ 0, -5,  7,  1,  1, -5],
    "colorado rockies":        [-6,  4,  6, -2,  4, -6],
    "los angeles dodgers":     [-4,  5, -5,  2,  0,  2],
    "pittsburgh pirates":      [-4, -1,  2,  1, 10, -9],
    "milwaukee brewers":       [ 1,  5,  0, -2, -1, -3],
    "seattle mariners":        [ 1,  1,  1,  0, -4,  2],
    "miami marlins":           [ 1, -4,  0, -1, -3,  6],
    "detroit tigers":          [-2, -3,  2,  5, -1, -1],
    "san francisco giants":    [-4, 10,  6, -1, -3, -6],
    "cincinnati reds":         [ 3,  4,  2,  0, -3, -5],
    "san diego padres":        [-4, -5,  6,  4,  3, -4],
    "philadelphia phillies":   [-5, -2,  1,  4,  2,  0],
    "st. louis cardinals":     [ 2,  4, -1,  0,  0, -4],
    "new york mets":           [-6,  2,  0, -2,  1,  5],
    "washington nationals":    [-9,  0,  1, -1,  6,  2],
    "minnesota twins":         [-8, -4,  5,  6, -1,  1],
    "new york yankees":        [-7, -1,  0, -1,  4,  4],
    "atlanta braves":          [ 2,  0, 11, -1, -7, -3],
    "texas rangers":           [-7,  2, -6,  1, -2, 11],
    "houston astros":          [ 0,  0,  0,  0,  0,  0],
}

# ── Ballpark dimensions ───────────────────────────────────────────────────────
PARK_INFO = {
    "los angeles angels":    {"park":"Angel Stadium",           "lf":330,"cf":400,"rf":330,"fence":18,"roof":"open",        "type":"neutral",  "note":"symmetrical; marine layer suppresses HRs"},
    "houston astros":        {"park":"Minute Maid Park",        "lf":315,"cf":435,"rf":326,"fence": 7,"roof":"retractable", "type":"pitcher",  "note":"deepest CF in MLB; short LF porch"},
    "oakland athletics":     {"park":"Oakland Coliseum",        "lf":330,"cf":400,"rf":330,"fence":18,"roof":"open",        "type":"pitcher",  "note":"largest foul territory in MLB"},
    "toronto blue jays":     {"park":"Rogers Centre",           "lf":328,"cf":400,"rf":328,"fence":10,"roof":"retractable", "type":"neutral",  "note":"turf boosts groundball singles"},
    "atlanta braves":        {"park":"Truist Park",             "lf":335,"cf":400,"rf":325,"fence": 6,"roof":"open",        "type":"hitter",   "note":"warm humid air; June winds boost offense"},
    "milwaukee brewers":     {"park":"American Family Field",   "lf":344,"cf":400,"rf":345,"fence": 8,"roof":"retractable", "type":"pitcher",  "note":"large dimensions suppress HRs"},
    "st. louis cardinals":   {"park":"Busch Stadium",           "lf":336,"cf":400,"rf":335,"fence": 8,"roof":"open",        "type":"neutral",  "note":"natural grass; moderate dimensions"},
    "chicago cubs":          {"park":"Wrigley Field",           "lf":355,"cf":400,"rf":353,"fence":11,"roof":"open",        "type":"variable", "note":"wind hugely variable; extreme hitter or pitcher park"},
    "arizona diamondbacks":  {"park":"Chase Field",             "lf":330,"cf":407,"rf":335,"fence": 7,"roof":"retractable", "type":"hitter",   "note":"high altitude (1,082 ft)"},
    "los angeles dodgers":   {"park":"Dodger Stadium",          "lf":330,"cf":395,"rf":330,"fence":14,"roof":"open",        "type":"pitcher",  "note":"marine layer suppresses carry; spacious foul territory"},
    "san francisco giants":  {"park":"Oracle Park",             "lf":339,"cf":399,"rf":309,"fence":25,"roof":"open",        "type":"pitcher",  "note":"marine layer; bay wind tricky in RF"},
    "cleveland guardians":   {"park":"Progressive Field",       "lf":325,"cf":405,"rf":325,"fence":19,"roof":"open",        "type":"neutral",  "note":"below-average HR park"},
    "seattle mariners":      {"park":"T-Mobile Park",           "lf":331,"cf":401,"rf":326,"fence":14,"roof":"retractable", "type":"pitcher",  "note":"marine air; roof often closed"},
    "miami marlins":         {"park":"loanDepot park",          "lf":344,"cf":407,"rf":335,"fence":12,"roof":"retractable", "type":"neutral",  "note":"roof often closed; average park"},
    "new york mets":         {"park":"Citi Field",              "lf":335,"cf":408,"rf":330,"fence":12,"roof":"open",        "type":"pitcher",  "note":"sea-level; large dimensions"},
    "washington nationals":  {"park":"Nationals Park",          "lf":336,"cf":402,"rf":335,"fence": 8,"roof":"open",        "type":"neutral",  "note":"cold Aprils very suppressive (-9 pts)"},
    "baltimore orioles":     {"park":"Camden Yards",            "lf":333,"cf":400,"rf":318,"fence":25,"roof":"open",        "type":"hitter",   "note":"short RF; warm summers"},
    "san diego padres":      {"park":"Petco Park",              "lf":336,"cf":396,"rf":322,"fence": 8,"roof":"open",        "type":"pitcher",  "note":"marine layer; large dimensions"},
    "philadelphia phillies": {"park":"Citizens Bank Park",      "lf":329,"cf":401,"rf":330,"fence": 6,"roof":"open",        "type":"hitter",   "note":"high HR rate historically"},
    "pittsburgh pirates":    {"park":"PNC Park",                "lf":325,"cf":399,"rf":320,"fence":21,"roof":"open",        "type":"pitcher",  "note":"high RF wall; spacious"},
    "texas rangers":         {"park":"Globe Life Field",        "lf":334,"cf":407,"rf":326,"fence": 8,"roof":"retractable", "type":"hitter",   "note":"1,200-ft altitude; September roof open adds runs"},
    "tampa bay rays":        {"park":"Tropicana Field",         "lf":315,"cf":404,"rf":322,"fence":10,"roof":"dome",        "type":"neutral",  "note":"dome; artificial turf"},
    "boston red sox":        {"park":"Fenway Park",             "lf":310,"cf":420,"rf":302,"fence":37,"roof":"open",        "type":"hitter",   "note":"Green Monster 37ft LF wall"},
    "cincinnati reds":       {"park":"Great American Ball Park","lf":328,"cf":404,"rf":325,"fence": 9,"roof":"open",        "type":"hitter",   "note":"one of the top HR parks in MLB"},
    "colorado rockies":      {"park":"Coors Field",             "lf":347,"cf":415,"rf":350,"fence": 8,"roof":"open",        "type":"hitter",   "note":"5,200 ft altitude; most extreme hitter park"},
    "kansas city royals":    {"park":"Kauffman Stadium",        "lf":330,"cf":410,"rf":330,"fence": 9,"roof":"open",        "type":"pitcher",  "note":"large dimensions; spray park"},
    "detroit tigers":        {"park":"Comerica Park",           "lf":345,"cf":420,"rf":330,"fence":14,"roof":"open",        "type":"pitcher",  "note":"very deep CF historically"},
    "minnesota twins":       {"park":"Target Field",            "lf":339,"cf":404,"rf":328,"fence": 8,"roof":"open",        "type":"neutral",  "note":"cold Aprils suppress offense (-8 pts)"},
    "chicago white sox":     {"park":"Guaranteed Rate Field",   "lf":330,"cf":400,"rf":335,"fence": 8,"roof":"open",        "type":"neutral",  "note":"July is extreme hitter month (+12 pts)"},
    "new york yankees":      {"park":"Yankee Stadium",          "lf":318,"cf":408,"rf":314,"fence": 8,"roof":"open",        "type":"hitter",   "note":"short porches both sides"},
}

# ── Lookup helpers ────────────────────────────────────────────────────────────

def _fuzzy_match(team: str, lookup: dict):
    """Exact match first, then partial."""
    t = clean_name(team)
    if t in lookup:
        return lookup[t]
    for key in lookup:
        if key in t or t in key:
            return lookup[key]
    return None


def park_run_mult(home_team: str, today: date = None) -> float:
    """Annual park factor × monthly deviation multiplier."""
    if today is None:
        today = date.today()

    annual = _fuzzy_match(home_team, PARK_FACTORS) or 1.0

    month = today.month
    month_idx = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}.get(month)
    if month_idx is None:
        return annual

    devs = _fuzzy_match(home_team, MONTHLY_DEV)
    if devs is None:
        return annual

    monthly_adj = 1 + (devs[month_idx] / 100)
    return annual * monthly_adj


def park_info(home_team: str) -> dict:
    return _fuzzy_match(home_team, PARK_INFO) or {}


def get_park_data() -> dict:
    print(f"[parks] loaded hardcoded 2025 park factors + monthly deviations for {len(PARK_FACTORS)} teams")
    return {"park_factors": PARK_FACTORS, "park_info": PARK_INFO, "monthly_dev": MONTHLY_DEV}
