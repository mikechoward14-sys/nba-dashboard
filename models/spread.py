"""
Spread model: projects expected point differential using team offensive/defensive ratings.

Correct formula (per-100-possessions ratings):
  home_pts = (home_OffRtg + away_DefRtg) / 2  * (game_pace / 100)
  away_pts = (away_OffRtg + home_DefRtg) / 2  * (game_pace / 100)

Intuition: average the offense's scoring rate with the defense's allowing rate
to get expected points per possession, then scale by pace.

Blended with recent form and home court adjustment.
"""
import pandas as pd
import numpy as np

HOME_COURT_BOOST = 2.5    # pts historically
RECENT_FORM_WEIGHT = 0.25  # blend weight for last-10 form vs season avg


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_row(df: pd.DataFrame, team_id: int):
    if df is None or df.empty or "TEAM_ID" not in df.columns:
        return None
    rows = df[df["TEAM_ID"] == team_id]
    return rows.iloc[0] if not rows.empty else None


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
    home_adv   = _get_row(team_advanced_df, home_team_id)
    away_adv   = _get_row(team_advanced_df, away_team_id)
    home_basic = _get_row(team_stats_df, home_team_id)
    away_basic = _get_row(team_stats_df, away_team_id)

    # ── Extract ratings (prefer advanced, fall back to basic per-game) ─────────
    # Advanced ratings are per-100-possessions; basic PTS/OPP_PTS are per-game.
    # We unify by converting both paths to per-game using pace.

    if home_adv is not None and "OFF_RATING" in home_adv.index and _safe_float(home_adv.get("OFF_RATING")) > 0:
        home_off_rtg = _safe_float(home_adv["OFF_RATING"])
        home_def_rtg = _safe_float(home_adv["DEF_RATING"])
        home_pace    = _safe_float(home_adv.get("PACE", 98.0))
        home_per100  = True
    elif home_basic is not None:
        # Basic stats already in per-game, treat as per-100 at league avg pace
        home_off_rtg = _safe_float(home_basic.get("PTS", 115.0))
        home_def_rtg = _safe_float(home_basic.get("OPP_PTS", 115.0))
        home_pace    = 98.0
        home_per100  = False
    else:
        home_off_rtg, home_def_rtg, home_pace, home_per100 = 114.0, 114.0, 98.0, True

    if away_adv is not None and "OFF_RATING" in away_adv.index and _safe_float(away_adv.get("OFF_RATING")) > 0:
        away_off_rtg = _safe_float(away_adv["OFF_RATING"])
        away_def_rtg = _safe_float(away_adv["DEF_RATING"])
        away_pace    = _safe_float(away_adv.get("PACE", 98.0))
        away_per100  = True
    elif away_basic is not None:
        away_off_rtg = _safe_float(away_basic.get("PTS", 115.0))
        away_def_rtg = _safe_float(away_basic.get("OPP_PTS", 115.0))
        away_pace    = 98.0
        away_per100  = False
    else:
        away_off_rtg, away_def_rtg, away_pace, away_per100 = 114.0, 114.0, 98.0, True

    # ── Project per-game scores ───────────────────────────────────────────────
    game_pace = (home_pace + away_pace) / 2.0

    if home_per100 and away_per100:
        # Both in per-100 ratings — apply pace scaling
        home_pts_season = (home_off_rtg + away_def_rtg) / 2.0 * (game_pace / 100.0)
        away_pts_season = (away_off_rtg + home_def_rtg) / 2.0 * (game_pace / 100.0)
    else:
        # At least one team fell back to per-game stats — average directly
        home_pts_season = (home_off_rtg + away_def_rtg) / 2.0
        away_pts_season = (away_off_rtg + home_def_rtg) / 2.0

    # ── Recent form adjustment ────────────────────────────────────────────────
    # PLUS_MINUS = team's point differential; positive means winning
    # A team with recent avg margin of +5 is scoring ~2.5 more than expected

    def recent_margin(log: pd.DataFrame) -> float:
        if log is None or log.empty or "PLUS_MINUS" not in log.columns:
            return 0.0
        return pd.to_numeric(log["PLUS_MINUS"], errors="coerce").dropna().mean()

    home_form = recent_margin(home_recent_df)
    away_form = recent_margin(away_recent_df)

    # Blend: season projection + fraction of recent form delta
    home_pts = home_pts_season + RECENT_FORM_WEIGHT * (home_form / 2.0)
    away_pts = away_pts_season + RECENT_FORM_WEIGHT * (away_form / 2.0)

    # ── Home court adjustment ─────────────────────────────────────────────────
    home_pts += HOME_COURT_BOOST / 2.0
    away_pts -= HOME_COURT_BOOST / 2.0

    margin = home_pts - away_pts
    spread_line = round(margin * 2) / 2  # nearest 0.5

    return {
        "spread": round(margin, 2),
        "spread_line": spread_line,
        "home_implied_pts": round(home_pts, 1),
        "away_implied_pts": round(away_pts, 1),
        "total": round(home_pts + away_pts, 1),
        "details": {
            "home_off_rtg":       round(home_off_rtg, 1),
            "home_def_rtg":       round(home_def_rtg, 1),
            "away_off_rtg":       round(away_off_rtg, 1),
            "away_def_rtg":       round(away_def_rtg, 1),
            "home_pace":          round(home_pace, 1),
            "away_pace":          round(away_pace, 1),
            "game_pace":          round(game_pace, 1),
            "home_recent_margin": round(home_form, 2),
            "away_recent_margin": round(away_form, 2),
            "home_court_boost":   HOME_COURT_BOOST,
        },
    }


def spread_to_cover_prob(spread: float) -> tuple[float, float]:
    """
    Probability each team covers using a logistic fit.
    Each point of spread ≈ 3% win-probability shift.
    """
    home_prob = 1 / (1 + 10 ** (-spread / 10.0))
    home_prob = max(0.05, min(0.95, home_prob))
    return round(home_prob, 3), round(1 - home_prob, 3)
