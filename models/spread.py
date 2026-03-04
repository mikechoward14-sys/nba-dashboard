"""
Spread model: projects expected point differential using team offensive/defensive ratings.

Approach:
  1. Pull season OffRtg and DefRtg for each team (per-100-possessions)
  2. Expected margin = (Home OffRtg - Away DefRtg) - (Away OffRtg - Home DefRtg)
     adjusted to per-game using pace
  3. Apply a home court boost (~2.5 pts historically)
  4. Blend with recent form (last 10 games avg margin)

All outputs are in points.
"""
import pandas as pd
import numpy as np

HOME_COURT_BOOST = 2.5   # pts
RECENT_FORM_WEIGHT = 0.35  # weight for last-10 vs season average
LEAGUE_AVG_PACE = 100.0   # possessions per 48 mins (approx)


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def expected_margin(
    home_team_id: int,
    away_team_id: int,
    team_stats_df: pd.DataFrame,
    team_advanced_df: pd.DataFrame,
    home_recent_df: pd.DataFrame,
    away_recent_df: pd.DataFrame,
) -> dict:
    """
    Returns a dict with:
      - spread: expected home margin (positive = home favored)
      - spread_line: rounded to nearest 0.5
      - home_implied_pts: projected home score
      - away_implied_pts: projected away score
      - total: implied total
      - details: breakdown dict for display
    """
    # ── Pull season ratings ──────────────────────────────────────────────────
    def get_row(df, team_id):
        if df.empty:
            return None
        mask = df["TEAM_ID"] == team_id
        rows = df[mask]
        return rows.iloc[0] if not rows.empty else None

    home_adv = get_row(team_advanced_df, home_team_id)
    away_adv = get_row(team_advanced_df, away_team_id)
    home_basic = get_row(team_stats_df, home_team_id)
    away_basic = get_row(team_stats_df, away_team_id)

    # OffRtg / DefRtg (advanced)
    if home_adv is not None and "OFF_RATING" in home_adv.index:
        home_off = _safe_float(home_adv.get("OFF_RATING"))
        home_def = _safe_float(home_adv.get("DEF_RATING"))
        home_pace = _safe_float(home_adv.get("PACE", LEAGUE_AVG_PACE))
    else:
        # Fallback: use basic PTS/OPP_PTS per game
        home_off = _safe_float(home_basic.get("PTS", 110)) if home_basic is not None else 110.0
        home_def = _safe_float(home_basic.get("OPP_PTS", 110)) if home_basic is not None else 110.0
        home_pace = LEAGUE_AVG_PACE

    if away_adv is not None and "OFF_RATING" in away_adv.index:
        away_off = _safe_float(away_adv.get("OFF_RATING"))
        away_def = _safe_float(away_adv.get("DEF_RATING"))
        away_pace = _safe_float(away_adv.get("PACE", LEAGUE_AVG_PACE))
    else:
        away_off = _safe_float(away_basic.get("PTS", 110)) if away_basic is not None else 110.0
        away_def = _safe_float(away_basic.get("OPP_PTS", 110)) if away_basic is not None else 110.0
        away_pace = LEAGUE_AVG_PACE

    # Shared pace (avg of both teams)
    game_pace = (home_pace + away_pace) / 2
    pace_adj = game_pace / 100.0

    league_avg_off = 112.0  # approx league average OffRtg 2024-25

    # Expected pts = (team OffRtg - opp DefRtg + league_avg) / 2 * pace_adj
    home_pts_season = ((home_off - away_def + league_avg_off) / 2) * pace_adj
    away_pts_season = ((away_off - home_def + league_avg_off) / 2) * pace_adj

    # ── Recent form adjustment ───────────────────────────────────────────────
    def recent_avg_margin(game_log: pd.DataFrame) -> float:
        if game_log.empty or "PLUS_MINUS" not in game_log.columns:
            return 0.0
        return game_log["PLUS_MINUS"].astype(float).mean()

    home_recent_margin = recent_avg_margin(home_recent_df)
    away_recent_margin = recent_avg_margin(away_recent_df)

    # Blend season and recent
    home_pts = home_pts_season + RECENT_FORM_WEIGHT * (home_recent_margin / 2)
    away_pts = away_pts_season + RECENT_FORM_WEIGHT * (-away_recent_margin / 2)

    # Home court
    home_pts += HOME_COURT_BOOST / 2
    away_pts -= HOME_COURT_BOOST / 2

    margin = home_pts - away_pts

    # Round spread to nearest 0.5
    spread_line = round(margin * 2) / 2

    return {
        "spread": round(margin, 2),
        "spread_line": spread_line,
        "home_implied_pts": round(home_pts, 1),
        "away_implied_pts": round(away_pts, 1),
        "total": round(home_pts + away_pts, 1),
        "details": {
            "home_off_rtg": round(home_off, 1),
            "home_def_rtg": round(home_def, 1),
            "away_off_rtg": round(away_off, 1),
            "away_def_rtg": round(away_def, 1),
            "home_pace": round(home_pace, 1),
            "away_pace": round(away_pace, 1),
            "home_recent_margin": round(home_recent_margin, 2),
            "away_recent_margin": round(away_recent_margin, 2),
            "home_court_boost": HOME_COURT_BOOST,
        },
    }


def spread_to_cover_prob(spread: float) -> tuple[float, float]:
    """
    Approximate probability each team covers using logistic curve.
    Returns (home_cover_prob, away_cover_prob).
    """
    # Each point of spread ≈ 3% probability shift
    pts_per_pct = 3.0
    home_prob = 0.5 + (spread / pts_per_pct) * 0.01
    home_prob = max(0.05, min(0.95, home_prob))
    return round(home_prob, 3), round(1 - home_prob, 3)
