"""
Player prop line generator.

For each player + stat category, we calculate:
  1. Season average (weighted toward recent)
  2. Last-N game rolling average
  3. Matchup adjustment: how well has opponent defended this position?
  4. Pace adjustment: fast/slow pace teams affect counting stats

Output: projected line, over prob, under prob
"""
import pandas as pd
import numpy as np
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
SEASON_WEIGHT = 0.4
RECENT_WEIGHT = 0.6
MATCHUP_ADJ_WEIGHT = 0.15


def _safe_series(game_log: pd.DataFrame, col: str) -> pd.Series:
    if game_log.empty or col not in game_log.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(game_log[col], errors="coerce").dropna()


def player_prop_line(
    player_game_log: pd.DataFrame,
    stat: str,
    opponent_stats: pd.Series | None = None,
    game_pace: float = 100.0,
    league_avg_pace: float = 100.0,
) -> dict:
    """
    Returns a dict with:
      - line: the projected over/under line (rounded to nearest 0.5)
      - projection: raw projected value
      - over_prob: probability of going over the line
      - under_prob: probability of going under the line
      - season_avg: full season average
      - recent_avg: last-10 average
      - std_dev: standard deviation of recent performances
      - hit_rate_over: % of last-20 games over line
      - games_played: number of games in log
    """
    series = _safe_series(player_game_log, stat)
    if series.empty:
        return _empty_result(stat)

    season_avg = series.mean()
    recent = series.iloc[:RECENT_WINDOW] if len(series) >= RECENT_WINDOW else series
    recent_avg = recent.mean()
    std_dev = series.std() if len(series) > 3 else recent_avg * 0.25

    # Blended projection
    projection = SEASON_WEIGHT * season_avg + RECENT_WEIGHT * recent_avg

    # Pace adjustment: faster pace → more possessions → more stats
    pace_factor = game_pace / league_avg_pace
    projection *= pace_factor

    # Matchup adjustment (optional)
    if opponent_stats is not None:
        opp_factor = _matchup_factor(stat, opponent_stats)
        projection = projection * (1 - MATCHUP_ADJ_WEIGHT) + projection * opp_factor * MATCHUP_ADJ_WEIGHT

    # Round to nearest 0.5
    line = round(projection * 2) / 2

    # Over/under probs using normal distribution
    if std_dev > 0:
        over_prob = float(1 - scipy_stats.norm.cdf(line + 0.5, loc=projection, scale=std_dev))
        under_prob = float(scipy_stats.norm.cdf(line - 0.5, loc=projection, scale=std_dev))
    else:
        over_prob = 0.5
        under_prob = 0.5

    # Historical hit rate vs the line
    hit_rate_over = float((series > line).mean()) if not series.empty else 0.5

    return {
        "stat": stat,
        "stat_label": PROP_CATEGORIES.get(stat, stat),
        "line": line,
        "projection": round(projection, 2),
        "over_prob": round(over_prob, 3),
        "under_prob": round(under_prob, 3),
        "season_avg": round(season_avg, 2),
        "recent_avg": round(recent_avg, 2),
        "std_dev": round(std_dev, 2),
        "hit_rate_over": round(hit_rate_over, 3),
        "games_played": len(series),
    }


def _matchup_factor(stat: str, opp_stats: pd.Series) -> float:
    """
    Returns a multiplier for the opponent's defensive impact on the stat.
    > 1.0 means opponent gives up more of this stat (favorable matchup)
    < 1.0 means opponent is stingy (unfavorable matchup)
    """
    # Map stat to relevant opponent defensive columns
    stat_to_opp = {
        "PTS": "OPP_PTS",
        "REB": "OPP_REB",
        "AST": "OPP_AST",
        "FG3M": "OPP_FG3M",
    }
    opp_col = stat_to_opp.get(stat)
    if opp_col is None or opp_col not in opp_stats.index:
        return 1.0

    league_avg = {
        "OPP_PTS": 112.0,
        "OPP_REB": 43.5,
        "OPP_AST": 25.0,
        "OPP_FG3M": 13.5,
    }
    avg = league_avg.get(opp_col, 1.0)
    opp_val = float(opp_stats[opp_col])
    if avg == 0:
        return 1.0
    return opp_val / avg


def _empty_result(stat: str) -> dict:
    return {
        "stat": stat,
        "stat_label": PROP_CATEGORIES.get(stat, stat),
        "line": 0.0,
        "projection": 0.0,
        "over_prob": 0.5,
        "under_prob": 0.5,
        "season_avg": 0.0,
        "recent_avg": 0.0,
        "std_dev": 0.0,
        "hit_rate_over": 0.5,
        "games_played": 0,
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
) -> list[dict]:
    if categories is None:
        categories = list(PROP_CATEGORIES.keys())
    results = []
    for stat in categories:
        result = player_prop_line(
            player_game_log, stat, opponent_stats, game_pace, league_avg_pace
        )
        if result["games_played"] > 0:
            result["ml_over"] = prob_to_moneyline(result["over_prob"])
            result["ml_under"] = prob_to_moneyline(result["under_prob"])
            results.append(result)
    return results
