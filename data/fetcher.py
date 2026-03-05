"""
NBA data fetcher using nba_api with local disk caching to avoid rate limits.
Pulls from NBA.com directly — always current season data.
All functions return pandas DataFrames.
"""
import os
import json
import time
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    playergamelog,
    scoreboardv2,
    leaguedashteamstats,
    leaguedashplayerstats,
    leaguehustlestatsplayer,
    leaguehustlestatsteam,
    commonteamroster,
)
from nba_api.stats.static import teams as nba_teams_static, players as nba_players_static

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

CURRENT_SEASON = "2025-26"

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

def _read_cache(key: str, ttl_hours: float = 6) -> pd.DataFrame | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) / 3600 > ttl_hours:
        return None
    with open(p) as f:
        return pd.DataFrame(json.load(f))

def _write_cache(key: str, df: pd.DataFrame):
    with open(_cache_path(key), "w") as f:
        json.dump(df.to_dict(orient="records"), f, default=str)

def _cached(key: str, ttl_hours: float, fn) -> pd.DataFrame:
    cached = _read_cache(key, ttl_hours)
    if cached is not None:
        return cached
    result = fn()
    if result is not None and not result.empty:
        _write_cache(key, result)
    return result if result is not None else pd.DataFrame()

def _fetch_with_retry(fn, retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            time.sleep(delay)
            return fn()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
    return pd.DataFrame()

# ── Static lookups ────────────────────────────────────────────────────────────

def get_all_teams() -> pd.DataFrame:
    return pd.DataFrame(nba_teams_static.get_teams())

def get_team_id(team_name: str) -> int | None:
    teams = get_all_teams()
    mask = teams["full_name"].str.lower().str.contains(team_name.lower())
    row = teams[mask]
    return int(row.iloc[0]["id"]) if not row.empty else None

def get_team_abbreviation(team_id: int) -> str | None:
    teams = get_all_teams()
    row = teams[teams["id"] == team_id]
    return row.iloc[0]["abbreviation"] if not row.empty else None

def get_all_active_players() -> pd.DataFrame:
    return pd.DataFrame(nba_players_static.get_active_players())

# ── Today's scoreboard ────────────────────────────────────────────────────────

def get_todays_games() -> pd.DataFrame:
    today = datetime.now().strftime("%m/%d/%Y")
    key = f"scoreboard_{today}"
    cached = _read_cache(key, ttl_hours=0.25)
    if cached is not None:
        return cached
    df = _fetch_with_retry(lambda: scoreboardv2.ScoreboardV2(game_date=today).game_header.get_data_frame())
    if not df.empty:
        _write_cache(key, df)
    return df

# ── Team season stats — all measure types ────────────────────────────────────

def _fetch_team_measure(season: str, measure: str) -> pd.DataFrame:
    return _fetch_with_retry(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense=measure,
        per_mode_detailed="PerGame",
        last_n_games=0,
    ).get_data_frames()[0])

def get_team_season_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    return _cached(f"team_base_{season}", 12, lambda: _fetch_team_measure(season, "Base"))

def get_team_advanced_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    return _cached(f"team_advanced_{season}", 12, lambda: _fetch_team_measure(season, "Advanced"))

def get_team_opponent_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """What each team gives up per game (defensive profile)."""
    return _cached(f"team_opponent_{season}", 12, lambda: _fetch_team_measure(season, "Opponent"))

def get_team_four_factors(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """EFG%, TOV%, OREB%, FT Rate — the four factors of winning."""
    return _cached(f"team_four_factors_{season}", 12, lambda: _fetch_team_measure(season, "Four Factors"))

def get_team_scoring_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Scoring distribution: paint, mid-range, 3PT breakdown."""
    return _cached(f"team_scoring_{season}", 12, lambda: _fetch_team_measure(season, "Scoring"))

def get_team_misc_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Second chance pts, pts off turnovers, fast break pts."""
    return _cached(f"team_misc_{season}", 12, lambda: _fetch_team_measure(season, "Misc"))

def get_team_hustle_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Contested shots, deflections, box outs, loose balls."""
    return _cached(f"team_hustle_{season}", 12, lambda: _fetch_with_retry(
        lambda: leaguehustlestatsteam.LeagueHustleStatsTeam(season=season).get_data_frames()[0]
    ))

def get_all_team_stats_merged(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """All team stat types merged into one wide DataFrame."""
    key = f"team_all_merged_{season}"
    cached = _read_cache(key, ttl_hours=12)
    if cached is not None:
        return cached

    base = get_team_season_stats(season)
    advanced = get_team_advanced_stats(season)
    opponent = get_team_opponent_stats(season)
    four_f = get_team_four_factors(season)
    misc = get_team_misc_stats(season)
    hustle = get_team_hustle_stats(season)

    merged = base.copy()
    for df in [advanced, opponent, four_f, misc, hustle]:
        if df.empty:
            continue
        new_cols = ["TEAM_ID"] + [c for c in df.columns if c not in merged.columns]
        merged = merged.merge(df[new_cols], on="TEAM_ID", how="left")

    _write_cache(key, merged)
    return merged

# ── Team game log ─────────────────────────────────────────────────────────────

def get_team_game_log(team_id: int, season: str = CURRENT_SEASON, last_n: int = 20) -> pd.DataFrame:
    key = f"team_gamelog_{team_id}_{season}"
    df = _cached(key, 4, lambda: _fetch_with_retry(
        lambda: teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
    ))
    return df.head(last_n) if not df.empty else df

# ── Player season stats ───────────────────────────────────────────────────────

def _fetch_player_measure(season: str, measure: str) -> pd.DataFrame:
    return _fetch_with_retry(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense=measure,
        per_mode_detailed="PerGame",
        last_n_games=0,
    ).get_data_frames()[0])

def get_player_season_stats(season: str = CURRENT_SEASON, min_minutes: float = 15.0) -> pd.DataFrame:
    key = f"player_base_{season}_{min_minutes}"
    df = _cached(key, 12, lambda: _fetch_player_measure(season, "Base"))
    if not df.empty and "MIN" in df.columns:
        df = df[pd.to_numeric(df["MIN"], errors="coerce") >= min_minutes]
    return df

def get_player_advanced_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    return _cached(f"player_advanced_{season}", 12, lambda: _fetch_player_measure(season, "Advanced"))

def get_player_hustle_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    return _cached(f"player_hustle_{season}", 12, lambda: _fetch_with_retry(
        lambda: leaguehustlestatsplayer.LeagueHustleStatsPlayer(season=season).get_data_frames()[0]
    ))

def get_player_scoring_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    return _cached(f"player_scoring_{season}", 12, lambda: _fetch_player_measure(season, "Scoring"))

def get_all_player_stats_merged(season: str = CURRENT_SEASON, min_minutes: float = 15.0) -> pd.DataFrame:
    """Base + advanced + hustle player stats in one DataFrame."""
    key = f"player_all_merged_{season}_{min_minutes}"
    cached = _read_cache(key, ttl_hours=12)
    if cached is not None:
        return cached

    base = get_player_season_stats(season, min_minutes)
    advanced = get_player_advanced_stats(season)
    hustle = get_player_hustle_stats(season)

    merged = base.copy()
    for df in [advanced, hustle]:
        if df.empty:
            continue
        pid = "PLAYER_ID"
        new_cols = [pid] + [c for c in df.columns if c not in merged.columns]
        if pid in df.columns:
            merged = merged.merge(df[new_cols], on=pid, how="left")

    _write_cache(key, merged)
    return merged

# ── Player game log ───────────────────────────────────────────────────────────

def get_player_game_log(player_id: int, season: str = CURRENT_SEASON, last_n: int = 30) -> pd.DataFrame:
    key = f"player_gamelog_{player_id}_{season}"
    df = _cached(key, 4, lambda: _fetch_with_retry(
        lambda: playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    ))
    return df.head(last_n) if not df.empty else df

# ── Roster ────────────────────────────────────────────────────────────────────

def get_team_roster(team_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    return _cached(f"roster_{team_id}_{season}", 24, lambda: _fetch_with_retry(
        lambda: commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    ))

# ── Head-to-head ──────────────────────────────────────────────────────────────

def get_head_to_head(team_id: int, opponent_id: int, last_n_seasons: int = 3) -> pd.DataFrame:
    current_year = datetime.now().year
    frames = []
    for i in range(last_n_seasons):
        year = current_year - i
        season = f"{year-1}-{str(year)[2:]}"
        key = f"h2h_{team_id}_{opponent_id}_{season}"
        def fetch(s=season):
            return _fetch_with_retry(lambda: leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=s,
                season_type_nullable="Regular Season",
            ).get_data_frames()[0])
        df = _cached(key, 24, fetch)
        if not df.empty and "MATCHUP" in df.columns:
            opp_abbr = get_team_abbreviation(opponent_id)
            if opp_abbr:
                df = df[df["MATCHUP"].str.contains(opp_abbr, na=False)]
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── Season games (for Elo) ────────────────────────────────────────────────────

def get_season_games(season: str = CURRENT_SEASON) -> pd.DataFrame:
    return _cached(f"all_games_{season}", 4, lambda: _fetch_with_retry(
        lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]
    ))
