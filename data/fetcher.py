"""
NBA data fetcher using nba_api with local disk caching to avoid rate limits.
All functions return pandas DataFrames.
"""
import os
import json
import time
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    playergamelog,
    scoreboardv2,
    teamdashboardbygeneralsplits,
    playerdashboardbygeneralsplits,
    leaguedashteamstats,
    leaguedashplayerstats,
    commonteamroster,
    boxscoresummaryv2,
)
from nba_api.stats.static import teams as nba_teams_static, players as nba_players_static

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

def _read_cache(key: str, ttl_hours: float = 6) -> pd.DataFrame | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = (time.time() - p.stat().st_mtime) / 3600
    if age > ttl_hours:
        return None
    with open(p) as f:
        data = json.load(f)
    return pd.DataFrame(data)

def _write_cache(key: str, df: pd.DataFrame):
    with open(_cache_path(key), "w") as f:
        json.dump(df.to_dict(orient="records"), f, default=str)

def _cached(key: str, ttl_hours: float, fn):
    cached = _read_cache(key, ttl_hours)
    if cached is not None:
        return cached
    result = fn()
    _write_cache(key, result)
    return result

# ── Static lookups ───────────────────────────────────────────────────────────

def get_all_teams() -> pd.DataFrame:
    teams = nba_teams_static.get_teams()
    return pd.DataFrame(teams)

def get_team_id(team_name: str) -> int | None:
    teams = get_all_teams()
    mask = teams["full_name"].str.lower().str.contains(team_name.lower())
    row = teams[mask]
    return int(row.iloc[0]["id"]) if not row.empty else None

def get_all_active_players() -> pd.DataFrame:
    players = nba_players_static.get_active_players()
    return pd.DataFrame(players)

# ── Today's scoreboard ───────────────────────────────────────────────────────

def get_todays_games() -> pd.DataFrame:
    today = datetime.now().strftime("%m/%d/%Y")
    key = f"scoreboard_{today}"
    cached = _read_cache(key, ttl_hours=0.5)
    if cached is not None:
        return cached

    sb = scoreboardv2.ScoreboardV2(game_date=today)
    games = sb.game_header.get_data_frame()
    line_score = sb.line_score.get_data_frame()
    _write_cache(key, games)
    return games

# ── Team season stats ────────────────────────────────────────────────────────

def get_team_season_stats(season: str = "2024-25") -> pd.DataFrame:
    key = f"team_season_stats_{season}"
    def fetch():
        time.sleep(0.6)
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        return df
    return _cached(key, ttl_hours=12, fn=fetch)

def get_team_advanced_stats(season: str = "2024-25") -> pd.DataFrame:
    key = f"team_advanced_{season}"
    def fetch():
        time.sleep(0.6)
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            last_n_games=0,
        ).get_data_frames()[0]
        return df
    return _cached(key, ttl_hours=12, fn=fetch)

# ── Team game log (last N games) ─────────────────────────────────────────────

def get_team_game_log(team_id: int, season: str = "2024-25", last_n: int = 20) -> pd.DataFrame:
    key = f"team_gamelog_{team_id}_{season}"
    def fetch():
        time.sleep(0.6)
        df = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
        ).get_data_frames()[0]
        return df
    df = _cached(key, ttl_hours=6, fn=fetch)
    return df.head(last_n)

# ── Player season stats ──────────────────────────────────────────────────────

def get_player_season_stats(season: str = "2024-25", min_minutes: float = 20.0) -> pd.DataFrame:
    key = f"player_season_stats_{season}_{min_minutes}"
    def fetch():
        time.sleep(0.6)
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        return df
    df = _cached(key, ttl_hours=12, fn=fetch)
    if "MIN" in df.columns:
        df = df[df["MIN"] >= min_minutes]
    return df

# ── Player game log ──────────────────────────────────────────────────────────

def get_player_game_log(player_id: int, season: str = "2024-25", last_n: int = 20) -> pd.DataFrame:
    key = f"player_gamelog_{player_id}_{season}"
    def fetch():
        time.sleep(0.6)
        df = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
        ).get_data_frames()[0]
        return df
    df = _cached(key, ttl_hours=6, fn=fetch)
    return df.head(last_n)

# ── Roster ───────────────────────────────────────────────────────────────────

def get_team_roster(team_id: int, season: str = "2024-25") -> pd.DataFrame:
    key = f"roster_{team_id}_{season}"
    def fetch():
        time.sleep(0.6)
        df = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season,
        ).get_data_frames()[0]
        return df
    return _cached(key, ttl_hours=24, fn=fetch)

# ── Head-to-head ─────────────────────────────────────────────────────────────

def get_head_to_head(team_id: int, opponent_id: int, last_n_seasons: int = 3) -> pd.DataFrame:
    """Return last few seasons of games between two teams."""
    seasons = []
    current_year = datetime.now().year
    for i in range(last_n_seasons):
        year = current_year - i
        seasons.append(f"{year-1}-{str(year)[2:]}")

    frames = []
    for season in seasons:
        key = f"h2h_{team_id}_{opponent_id}_{season}"
        def fetch(s=season):
            time.sleep(0.6)
            df = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=s,
                season_type_nullable="Regular Season",
            ).get_data_frames()[0]
            return df
        df = _cached(key, ttl_hours=24, fn=fetch)
        # Filter to games vs opponent
        if not df.empty and "MATCHUP" in df.columns:
            opp_abbr = get_team_abbreviation(opponent_id)
            if opp_abbr:
                df = df[df["MATCHUP"].str.contains(opp_abbr, na=False)]
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def get_team_abbreviation(team_id: int) -> str | None:
    teams = get_all_teams()
    row = teams[teams["id"] == team_id]
    return row.iloc[0]["abbreviation"] if not row.empty else None

# ── Historical games (for Elo) ────────────────────────────────────────────────

def get_season_games(season: str = "2024-25") -> pd.DataFrame:
    key = f"all_games_{season}"
    def fetch():
        time.sleep(0.6)
        df = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
        ).get_data_frames()[0]
        return df
    return _cached(key, ttl_hours=6, fn=fetch)
