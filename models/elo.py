"""
Elo rating system for NBA teams.
Used to generate win probabilities for moneyline calculations.

Formula reference:
  Expected score: E_a = 1 / (1 + 10^((R_b - R_a) / 400))
  Updated rating: R_a' = R_a + K * (S_a - E_a)

Home court advantage = +100 Elo points (≈ 3.5 pts historically)
"""
import pandas as pd
import numpy as np

STARTING_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 100  # Elo points


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_ratings(games_df: pd.DataFrame) -> dict[int, float]:
    """
    Build Elo ratings from a season games DataFrame (from LeagueGameFinder).
    Returns {team_id: elo_rating}.

    Expected columns: TEAM_ID, GAME_DATE, WL, MATCHUP, PTS, PLUS_MINUS
    """
    if games_df.empty:
        return {}

    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    # Deduplicate: each game appears twice (once per team). Keep home team rows.
    # MATCHUP for home team looks like "BOS vs. MIA", away is "MIA @ BOS"
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ")
    home_games = df[df["IS_HOME"]].copy()

    ratings: dict[int, float] = {}

    for _, row in home_games.iterrows():
        home_id = int(row["TEAM_ID"])
        # Parse opponent from MATCHUP: "BOS vs. MIA" → "MIA"
        parts = row["MATCHUP"].split(" vs. ")
        opp_abbr = parts[1].strip() if len(parts) == 2 else None

        # Look up away team id from same game
        away_rows = df[
            (df["GAME_ID"] == row["GAME_ID"]) & (~df["IS_HOME"])
        ]
        if away_rows.empty:
            continue
        away_id = int(away_rows.iloc[0]["TEAM_ID"])

        r_home = ratings.get(home_id, STARTING_ELO)
        r_away = ratings.get(away_id, STARTING_ELO)

        # Add home advantage to home team for expected calc
        e_home = _expected(r_home + HOME_ADVANTAGE, r_away)
        e_away = 1.0 - e_home

        home_win = 1.0 if row["WL"] == "W" else 0.0
        away_win = 1.0 - home_win

        ratings[home_id] = r_home + K_FACTOR * (home_win - e_home)
        ratings[away_id] = r_away + K_FACTOR * (away_win - e_away)

    return ratings


def win_probability(
    home_elo: float,
    away_elo: float,
    neutral_site: bool = False,
) -> tuple[float, float]:
    """
    Returns (home_win_prob, away_win_prob).
    """
    adj_home = home_elo + (0 if neutral_site else HOME_ADVANTAGE)
    p_home = _expected(adj_home, away_elo)
    return round(p_home, 4), round(1 - p_home, 4)


def prob_to_moneyline(prob: float) -> int:
    """Convert win probability to American moneyline odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        ml = -round((prob / (1 - prob)) * 100)
    else:
        ml = round(((1 - prob) / prob) * 100)
    return int(ml)


def moneyline_to_implied_prob(ml: int) -> float:
    """Convert American moneyline to implied probability (no vig)."""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)
