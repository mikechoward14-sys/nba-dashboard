"""
Player prop line generator.

Key design: the model returns a projection + distribution (mean + std_dev).
Over/under probabilities are always calculated against the BOOK'S line,
not our projected line. This is what surfaces under bets correctly.

If DraftKings has PTS at 27.5 and our projection is 23.0 with std 5.0:
  P(over 27.5) = 1 - norm.cdf(27.5, 23, 5) ≈ 19% → lean Under
  P(under 27.5) = norm.cdf(27.5, 23, 5) ≈ 81% → strong Under edge

Calibration: the CALIBRATION_SCALE is a multiplier on std_dev derived from
historical hit/miss data. Starts at 1.0, auto-adjusted by the tracker.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

PROP_CATEGORIES = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "FG3M": "3-Pointers Made",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers",
}

RECENT_WINDOW = 10
SEASON_WEIGHT = 0.35
RECENT_WEIGHT = 0.65
MATCHUP_ADJ_WEIGHT = 0.20

CALIBRATION_FILE = Path(__file__).parent.parent / ".tracking" / "calibration.json"


def _load_calibration() -> dict:
    """Load per-stat calibration scale factors from tracker. Default 1.0."""
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            pass
    return {}


def _safe_series(game_log: pd.DataFrame, col: str) -> pd.Series:
    if game_log.empty or col not in game_log.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(game_log[col], errors="coerce").dropna()


def prob_at_line(projection: float, std_dev: float, book_line: float, calibration_scale: float = 1.0) -> tuple[float, float]:
    """
    Given our projection + std_dev, calculate P(over book_line) and P(under book_line).
    Uses a continuity correction of ±0.5 since NBA stats are integers.
    calibration_scale adjusts std_dev based on historical accuracy.
    """
    adj_std = max(std_dev * calibration_scale, 0.1)
    over_prob  = float(1 - scipy_stats.norm.cdf(book_line + 0.5, loc=projection, scale=adj_std))
    under_prob = float(scipy_stats.norm.cdf(book_line - 0.5, loc=projection, scale=adj_std))
    return round(max(0.01, min(0.99, over_prob)), 4), round(max(0.01, min(0.99, under_prob)), 4)


def player_prop_line(
    player_game_log: pd.DataFrame,
    stat: str,
    opponent_stats: pd.Series | None = None,
    game_pace: float = 100.0,
    league_avg_pace: float = 100.0,
    book_line: float | None = None,
) -> dict:
    """
    Returns projection dict. If book_line is provided, over/under probs are
    calculated against that line. Otherwise they're calculated against our line.
    Always returns both the over AND under direction accurately.
    """
    calibration = _load_calibration()
    cal_scale = calibration.get(stat, 1.0)

    series = _safe_series(player_game_log, stat)
    if series.empty:
        return _empty_result(stat)

    season_avg = series.mean()
    recent = series.iloc[:RECENT_WINDOW] if len(series) >= RECENT_WINDOW else series
    recent_avg = recent.mean()

    # Std dev: use full sample for stability, floor at 15% of mean
    std_dev = series.std() if len(series) > 4 else recent_avg * 0.30
    std_dev = max(std_dev, season_avg * 0.15)

    # Blended projection
    projection = SEASON_WEIGHT * season_avg + RECENT_WEIGHT * recent_avg

    # Pace adjustment
    if league_avg_pace > 0:
        projection *= (game_pace / league_avg_pace)

    # Matchup adjustment
    if opponent_stats is not None:
        opp_factor = _matchup_factor(stat, opponent_stats)
        # Blend: 80% projection, 20% matchup-adjusted
        projection = projection * (1 - MATCHUP_ADJ_WEIGHT) + projection * opp_factor * MATCHUP_ADJ_WEIGHT

    # Our fair line (rounded to nearest 0.5)
    our_line = round(projection * 2) / 2

    # Calculate probs against book line if available, else vs our line
    target_line = book_line if book_line is not None else our_line
    over_prob, under_prob = prob_at_line(projection, std_dev, target_line, cal_scale)

    # Historical hit rate vs our line (for chart reference)
    hit_rate_over = float((series > our_line).mean()) if not series.empty else 0.5
    # Historical hit rate vs book line
    hit_rate_over_book = float((series > target_line).mean()) if book_line is not None else hit_rate_over

    return {
        "stat": stat,
        "stat_label": PROP_CATEGORIES.get(stat, stat),
        "our_line": our_line,
        "line": our_line,  # kept for backward compat
        "book_line": book_line,
        "target_line": target_line,
        "projection": round(projection, 2),
        "std_dev": round(std_dev, 2),
        "over_prob": over_prob,
        "under_prob": under_prob,
        "season_avg": round(season_avg, 2),
        "recent_avg": round(recent_avg, 2),
        "hit_rate_over": round(hit_rate_over, 3),
        "hit_rate_over_book": round(hit_rate_over_book, 3),
        "games_played": len(series),
        "calibration_scale": cal_scale,
    }


def _matchup_factor(stat: str, opp_stats: pd.Series) -> float:
    """Multiplier based on opponent's defensive tendencies for this stat."""
    stat_to_opp = {
        "PTS": "OPP_PTS",
        "REB": "OPP_REB",
        "AST": "OPP_AST",
        "FG3M": "OPP_FG3M",
    }
    opp_col = stat_to_opp.get(stat)
    if opp_col is None or opp_col not in opp_stats.index:
        return 1.0
    league_avg = {"OPP_PTS": 115.0, "OPP_REB": 43.5, "OPP_AST": 25.0, "OPP_FG3M": 13.5}
    avg = league_avg.get(opp_col, 1.0)
    opp_val = float(opp_stats[opp_col])
    return opp_val / avg if avg > 0 else 1.0


def _empty_result(stat: str) -> dict:
    return {
        "stat": stat, "stat_label": PROP_CATEGORIES.get(stat, stat),
        "our_line": 0.0, "line": 0.0, "book_line": None, "target_line": 0.0,
        "projection": 0.0, "std_dev": 0.0,
        "over_prob": 0.5, "under_prob": 0.5,
        "season_avg": 0.0, "recent_avg": 0.0,
        "hit_rate_over": 0.5, "hit_rate_over_book": 0.5,
        "games_played": 0, "calibration_scale": 1.0,
    }


def prob_to_moneyline(prob: float) -> int:
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return -round((prob / (1 - prob)) * 100)
    return round(((1 - prob) / prob) * 100)


def all_props_for_player(
    player_game_log: pd.DataFrame,
    opponent_stats: pd.Series | None = None,
    game_pace: float = 100.0,
    league_avg_pace: float = 100.0,
    categories: list[str] | None = None,
    book_props: dict | None = None,
) -> list[dict]:
    """
    Generate props for all categories. If book_props provided ({stat: {line, ...}}),
    uses book lines for probability calculations so unders surface correctly.
    """
    if categories is None:
        categories = list(PROP_CATEGORIES.keys())
    results = []
    for stat in categories:
        bk_line = None
        if book_props and stat in book_props and book_props[stat].get("line"):
            bk_line = float(book_props[stat]["line"])

        result = player_prop_line(
            player_game_log, stat, opponent_stats, game_pace, league_avg_pace,
            book_line=bk_line,
        )
        if result["games_played"] > 0:
            result["ml_over"]  = prob_to_moneyline(result["over_prob"])
            result["ml_under"] = prob_to_moneyline(result["under_prob"])
            results.append(result)
    return results
