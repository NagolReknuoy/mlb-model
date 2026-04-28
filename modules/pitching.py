# pitching.py — Probable starters + Statcast metrics (2-year blended)

import requests
import pandas as pd
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings("ignore")

from pybaseball import statcast_pitcher
from pybaseball import cache
cache.enable()   # cache Statcast pulls so repeat runs are instant

from modules.utils import clean_name, safe_mean


# ── MLB Stats API probable pitcher fetch ─────────────────────────────────────

def _get_all_probables(date_str: str) -> pd.DataFrame:
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_str}&hydrate=probablePitcher,team"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        dates = data.get("dates", [])
        if not dates:
            return pd.DataFrame()

        rows = []
        for game in dates[0].get("games", []):
            home_pp = game.get("teams", {}).get("home", {}).get("probablePitcher")
            away_pp = game.get("teams", {}).get("away", {}).get("probablePitcher")
            rows.append({
                "game_pk":       game["gamePk"],
                "home_pitcher":  clean_name(home_pp["fullName"]) if home_pp else None,
                "away_pitcher":  clean_name(away_pp["fullName"]) if away_pp else None,
                "home_mlbam_id": int(home_pp["id"]) if home_pp else None,
                "away_mlbam_id": int(away_pp["id"]) if away_pp else None,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[pitching] MLB API error: {e}")
        return pd.DataFrame()


# ── Statcast fetch for one pitcher one season ─────────────────────────────────

def _fetch_statcast(mlbam_id: int, season: int, today: date) -> pd.DataFrame:
    start = f"{season}-03-01"
    end   = str(today) if season == today.year else f"{season}-11-01"
    try:
        df = statcast_pitcher(start, end, player_id=mlbam_id)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Summarise raw Statcast rows into per-pitcher metrics ──────────────────────

def _summarise_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["player_name","velo","kbb","era","ppg"])

    df = df.copy()
    df["player_name"] = df["player_name"].apply(
        lambda x: clean_name(" ".join(reversed(x.split(", ")))) if isinstance(x, str) and "," in x else clean_name(str(x))
    )

    results = []
    for name, grp in df.groupby("player_name"):
        velo   = grp["release_speed"].mean()
        k      = (grp["events"] == "strikeout").sum()
        bb     = (grp["events"] == "walk").sum()
        hr     = (grp["events"] == "home_run").sum()
        pa     = len(grp)
        games  = grp["game_pk"].nunique() if "game_pk" in grp.columns else 1
        
        # K% and BB% relative to plate appearances (more stable than K/BB ratio)
        k_pct  = k / max(pa, 1)
        bb_pct = bb / max(pa, 1)
        hr_pct = hr / max(pa, 1)
        
        # K/BB ratio (capped to prevent extreme values)
        kbb = min(k / bb, 8.0) if bb > 0 else 4.0  # 4.0 = good default when no walks
        
        # Estimated ERA proxy using FIP components (no fielding needed)
        # FIP = (13*HR + 3*BB - 2*K) / IP + constant
        # We approximate IP as PA/4.3 (avg PA per inning for starters)
        ip_approx = pa / 4.3
        if ip_approx > 0:
            fip_raw = (13 * hr + 3 * bb - 2 * k) / ip_approx
            # Normalize to ERA scale (add ~3.2 constant)
            era_proxy = fip_raw + 3.2
        else:
            era_proxy = 4.50  # league average default
            
        ppg = pa / max(1, games)
        results.append({
            "player_name": name,
            "velo":      velo,
            "kbb":       kbb,
            "k_pct":     k_pct,
            "bb_pct":    bb_pct,
            "era_proxy": era_proxy,
            "ppg":       ppg,
        })
    return pd.DataFrame(results)


# ── Pitch factor multiplier ───────────────────────────────────────────────────

def pitch_factor(velo, kbb, whip, lg_velo, lg_kbb,
                 era_proxy=None, lg_era=None) -> float:
    """
    Run suppression multiplier. < 1 = better than league (fewer runs).
    
    Uses FIP-based ERA proxy as primary metric (60% weight) since it
    directly estimates runs allowed. K/BB secondary (40% weight).
    Velo removed — too noisy when mixing starter/reliever Statcast data.
    
    Result clamped to 0.85-1.15 (±15% max deviation per pitcher).
    League average pitcher = 1.0, ace = ~0.87, replacement level = ~1.13.
    """
    factors = []

    
    lg_era_val = lg_era if (lg_era and not pd.isna(lg_era)) else 4.30
    if era_proxy is not None and not pd.isna(era_proxy) and era_proxy > 0:
        # Higher ERA = more runs, so era/lg_era > 1 means worse pitcher
        era_f = (era_proxy / lg_era_val) ** 0.6
        factors.append(era_f)

    # K/BB component (40% weight)
    if (not pd.isna(kbb) and not pd.isna(lg_kbb)
            and kbb > 0 and lg_kbb > 0):
        kbb_f = (lg_kbb / kbb) ** 0.4
        factors.append(kbb_f)

    if not factors:
        return 1.0

    result = 1.0
    for f in factors:
        result *= f

    # Hard clamp ±15%
    return max(0.85, min(1.15, result))


def starter_ip(ppg) -> float:
    if pd.isna(ppg):
        return 5.5
    return min(7.0, max(4.0, ppg / 15))


def blend_starter_bullpen(starter_mult: float, ip: float, bullpen_mult: float = 1.03) -> float:
    share = min(1.0, max(0.0, ip / 9))
    return np.exp(share * np.log(starter_mult) + (1 - share) * np.log(bullpen_mult))


# ── Main function ─────────────────────────────────────────────────────────────

def get_pitching_data(today_games: pd.DataFrame, today: date = None) -> dict:
    if today is None:
        today = date.today()

    date_str = str(today)
    print(f"[pitching] fetching probable starters for {date_str} ...")

    probables_raw = _get_all_probables(date_str)

    # Merge onto today's games
    if probables_raw.empty:
        probables = today_games.copy()
        for col in ["home_pitcher","away_pitcher"]:
            probables[col] = None
        for col in ["home_mlbam_id","away_mlbam_id"]:
            probables[col] = np.nan
    else:
        probables = today_games.merge(probables_raw, on="game_pk", how="left")

    has_pitchers = probables["home_pitcher"].notna().any() or probables["away_pitcher"].notna().any()

    empty_lg = {"lg_velo": np.nan, "lg_kbb": np.nan}
    stat_cols = ["home_velo","home_kbb","home_whip","home_ppg",
                 "away_velo","away_kbb","away_whip","away_ppg"]

    if not has_pitchers:
        print("[pitching] no probable starters announced yet")
        for col in stat_cols:
            probables[col] = np.nan
        return {"pitchers_df": probables, "league_pitch": empty_lg}

    # Build unique pitcher list with MLBAM IDs from the API
    home_p = probables[["home_pitcher","home_mlbam_id"]].rename(
        columns={"home_pitcher":"name","home_mlbam_id":"mlbam_id"})
    away_p = probables[["away_pitcher","away_mlbam_id"]].rename(
        columns={"away_pitcher":"name","away_mlbam_id":"mlbam_id"})
    pitcher_list = pd.concat([home_p, away_p]).dropna(subset=["name"]).drop_duplicates("name").reset_index(drop=True)
    pitcher_list["mlbam_id"] = pitcher_list["mlbam_id"].astype("Int64")

    n = len(pitcher_list)
    print(f"[pitching] found {n} probable pitchers — pulling Statcast (cached after first run) ...")

    cur_yr  = today.year
    prev_yr = cur_yr - 1

    all_cur, all_prev = [], []
    for _, row in pitcher_list.iterrows():
        mid = row["mlbam_id"]
        if pd.isna(mid):
            continue
        mid = int(mid)
        df_cur  = _fetch_statcast(mid, cur_yr,  today)
        df_prev = _fetch_statcast(mid, prev_yr, today)
        if not df_cur.empty:
            all_cur.append(df_cur)
        if not df_prev.empty:
            all_prev.append(df_prev)

    df_cur_combined  = pd.concat(all_cur,  ignore_index=True) if all_cur  else pd.DataFrame()
    df_prev_combined = pd.concat(all_prev, ignore_index=True) if all_prev else pd.DataFrame()

    n_cur  = len(df_cur_combined)
    n_prev = len(df_prev_combined)
    print(f"[pitching] Statcast rows — {cur_yr}: {n_cur} | {prev_yr}: {n_prev}")

    # Blend: 2× weight on current season for recency
    blended = pd.concat([
        pd.concat([df_cur_combined, df_cur_combined], ignore_index=True) if n_cur  > 0 else pd.DataFrame(),
        df_prev_combined if n_prev > 0 else pd.DataFrame()
    ], ignore_index=True)

    if blended.empty:
        print("[pitching] no Statcast data – using neutral pitcher factors")
        pitch_form = pd.DataFrame(columns=["player_name","velo","kbb","whip","ppg"])
        lg = empty_lg
    else:
        pitch_form = _summarise_pitchers(blended)
        lg = {
            "lg_velo": pitch_form["velo"].mean(),
            "lg_kbb":  pitch_form["kbb"].mean(),
            "lg_era":  pitch_form["era_proxy"].mean(),
        }
        src = (f"{cur_yr}(2×)+{prev_yr}(1×)" if n_cur > 0 and n_prev > 0
               else f"{cur_yr} only" if n_cur > 0 else f"{prev_yr} only")
        print(f"[pitching] {len(pitch_form)} pitchers | {src} | "
              f"lg velo {lg['lg_velo']:.1f} | lg K/BB {lg['lg_kbb']:.2f}")

    # Join stats onto probables
    def _join(df, side):
        name_col = f"{side}_pitcher"
        merged = df.merge(
            pitch_form.rename(columns={
                "player_name":  name_col,
                "velo":         f"{side}_velo",
                "kbb":          f"{side}_kbb",
                "era_proxy":    f"{side}_era",
                "ppg":          f"{side}_ppg",
            }),
            on=name_col, how="left"
        )
        return merged

    probables = _join(probables, "home")
    probables = _join(probables, "away")

    stat_cols = ["home_velo","home_kbb","home_era","home_ppg",
                 "away_velo","away_kbb","away_era","away_ppg"]
    for col in stat_cols:
        if col not in probables.columns:
            probables[col] = np.nan

    return {"pitchers_df": probables, "league_pitch": lg}
