"""
Odds fetcher — pulls live lines from The Odds API.
Free tier: 500 requests/month. We cache aggressively (30 min) to conserve quota.

API docs: https://the-odds-api.com/liveapi/guides/v4/
Free key: https://the-odds-api.com
"""
import os
import json
import time
import hashlib
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "odds"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Preferred books in priority order for display
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "bovada"]

# Map Odds API player prop market keys → our stat column names
PROP_MARKET_MAP = {
    "player_points":           "PTS",
    "player_rebounds":         "REB",
    "player_assists":          "AST",
    "player_threes":           "FG3M",
    "player_steals":           "STL",
    "player_blocks":           "BLK",
    "player_turnovers":        "TOV",
    "player_points_rebounds_assists": "PRA",
    "player_points_rebounds":  "PR",
    "player_points_assists":   "PA",
}

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

def _read_cache(key: str, ttl_min: float = 30) -> dict | list | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) / 60 > ttl_min:
        return None
    with open(p) as f:
        return json.load(f)

def _write_cache(key: str, data):
    with open(_cache_path(key), "w") as f:
        json.dump(data, f, default=str)

# ── Core fetcher ──────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, api_key: str) -> list | dict | None:
    params["apiKey"] = api_key
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
        remaining = r.headers.get("x-requests-remaining", "?")
        if r.status_code == 401:
            return {"error": "Invalid API key"}
        if r.status_code == 422:
            return {"error": "Invalid request parameters"}
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# ── Game odds (moneyline, spread, total) ──────────────────────────────────────

def get_game_odds(api_key: str) -> list[dict]:
    """
    Returns all NBA game odds for today.
    Each item: {home_team, away_team, commence_time, bookmakers: [...]}
    """
    if not api_key:
        return []
    key = f"game_odds_{datetime.now().strftime('%Y%m%d%H')}"
    cached = _read_cache(key, ttl_min=30)
    if cached is not None:
        return cached

    data = _get(
        f"sports/{SPORT}/odds",
        {"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"},
        api_key,
    )
    if isinstance(data, dict) and "error" in data:
        return [{"error": data["error"]}]
    result = data or []
    _write_cache(key, result)
    return result


def parse_game_odds(odds_data: list[dict]) -> dict[str, dict]:
    """
    Returns {"{away} @ {home}": {home_ml, away_ml, spread, total, book}}
    Using best available book from PREFERRED_BOOKS.
    """
    result = {}
    for game in odds_data:
        if "error" in game:
            continue
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        matchup_key = f"{away} @ {home}"

        lines = {"home_team": home, "away_team": away,
                 "home_ml": None, "away_ml": None,
                 "home_spread": None, "away_spread": None,
                 "total": None, "book": None}

        # Try preferred books first, then any
        books = game.get("bookmakers", [])
        ordered = sorted(books, key=lambda b: (
            PREFERRED_BOOKS.index(b["key"]) if b["key"] in PREFERRED_BOOKS else 999
        ))

        for book in ordered:
            book_name = book.get("title", book.get("key", ""))
            for market in book.get("markets", []):
                mkt = market.get("key")
                outcomes = market.get("outcomes", [])

                if mkt == "h2h":
                    for o in outcomes:
                        if o["name"] == home:
                            lines["home_ml"] = lines["home_ml"] or int(o["price"])
                        elif o["name"] == away:
                            lines["away_ml"] = lines["away_ml"] or int(o["price"])

                elif mkt == "spreads":
                    for o in outcomes:
                        if o["name"] == home:
                            lines["home_spread"] = lines["home_spread"] or float(o.get("point", 0))
                        elif o["name"] == away:
                            lines["away_spread"] = lines["away_spread"] or float(o.get("point", 0))

                elif mkt == "totals":
                    for o in outcomes:
                        if o["name"] == "Over":
                            lines["total"] = lines["total"] or float(o.get("point", 0))

            if all(v is not None for v in [lines["home_ml"], lines["away_ml"], lines["home_spread"], lines["total"]]):
                lines["book"] = book_name
                break

        result[matchup_key] = lines
    return result


def find_game_odds(market_lines: dict, home_team: str, away_team: str) -> dict | None:
    """
    Fuzzy match team names from NBA API to Odds API game keys.
    Returns the matched lines dict or None.
    """
    for key, lines in market_lines.items():
        if _team_match(lines.get("home_team", ""), home_team) and \
           _team_match(lines.get("away_team", ""), away_team):
            return lines
    return None


def _team_match(odds_name: str, nba_name: str) -> bool:
    """Match e.g. 'Boston Celtics' == 'Boston Celtics' or partial last-word match."""
    if not odds_name or not nba_name:
        return False
    if odds_name.lower() == nba_name.lower():
        return True
    # Match on last word (city or nickname)
    odds_word = odds_name.split()[-1].lower()
    nba_word = nba_name.split()[-1].lower()
    return odds_word == nba_word

# ── Player props ──────────────────────────────────────────────────────────────

def get_event_ids(api_key: str) -> list[dict]:
    """Get today's NBA event IDs needed to fetch player props."""
    if not api_key:
        return []
    key = f"events_{datetime.now().strftime('%Y%m%d%H')}"
    cached = _read_cache(key, ttl_min=60)
    if cached is not None:
        return cached

    data = _get(f"sports/{SPORT}/events", {"regions": "us"}, api_key)
    if not isinstance(data, list):
        return []
    _write_cache(key, data)
    return data


def get_player_props_for_event(event_id: str, api_key: str) -> dict:
    """
    Fetch all player prop markets for a single event.
    Returns {player_name: {stat: {line, over_ml, under_ml, book}}}
    """
    if not api_key:
        return {}
    markets = ",".join(PROP_MARKET_MAP.keys())
    key = f"props_{event_id}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = _read_cache(key, ttl_min=30)
    if cached is not None:
        return cached

    data = _get(
        f"sports/{SPORT}/events/{event_id}/odds",
        {"regions": "us", "markets": markets, "oddsFormat": "american"},
        api_key,
    )
    if not isinstance(data, dict) or "bookmakers" not in data:
        return {}

    result = {}  # {player_name: {stat_col: {line, over_ml, under_ml, book}}}
    books = data.get("bookmakers", [])
    ordered = sorted(books, key=lambda b: (
        PREFERRED_BOOKS.index(b["key"]) if b["key"] in PREFERRED_BOOKS else 999
    ))

    for book in ordered:
        book_name = book.get("title", book.get("key", ""))
        for market in book.get("markets", []):
            mkt_key = market.get("key", "")
            stat_col = PROP_MARKET_MAP.get(mkt_key)
            if not stat_col:
                continue

            outcomes = market.get("outcomes", [])
            # outcomes come in pairs: {name, description, price, point}
            # description = player name, name = Over/Under
            player_lines: dict[str, dict] = {}
            for o in outcomes:
                player = o.get("description", "")
                side = o.get("name", "")
                price = int(o.get("price", 0))
                point = float(o.get("point", 0))
                if not player:
                    continue
                if player not in player_lines:
                    player_lines[player] = {"line": point, "over_ml": None, "under_ml": None, "book": book_name}
                if side == "Over":
                    player_lines[player]["over_ml"] = price
                elif side == "Under":
                    player_lines[player]["under_ml"] = price

            for player, pdata in player_lines.items():
                if player not in result:
                    result[player] = {}
                if stat_col not in result[player]:
                    result[player][stat_col] = pdata

    _write_cache(key, result)
    return result


def get_all_player_props(api_key: str) -> dict[str, dict]:
    """
    Returns props for all of today's games merged.
    {player_name: {stat_col: {line, over_ml, under_ml, book}}}
    """
    if not api_key:
        return {}
    events = get_event_ids(api_key)
    all_props = {}
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        props = get_player_props_for_event(event_id, api_key)
        for player, stats in props.items():
            if player not in all_props:
                all_props[player] = {}
            all_props[player].update(stats)
    return all_props


def find_player_props(all_props: dict, player_name: str) -> dict:
    """Fuzzy match player name and return their prop lines."""
    if not all_props:
        return {}
    # Exact match first
    if player_name in all_props:
        return all_props[player_name]
    # Last name match
    last = player_name.split()[-1].lower()
    for name, props in all_props.items():
        if name.split()[-1].lower() == last:
            return props
    return {}
