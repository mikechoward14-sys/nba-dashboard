"""
Today's Games page — all NBA games today with our lines vs sportsbook lines.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from data.fetcher import (
    get_todays_games,
    get_team_season_stats,
    get_team_advanced_stats,
    get_team_game_log,
    get_season_games,
    get_all_teams,
)
from data.odds_fetcher import get_game_odds, parse_game_odds, find_game_odds
from models.elo import compute_elo_ratings, win_probability, prob_to_moneyline
from models.spread import expected_margin
from utils.formatting import fmt_moneyline, spread_display
from utils.line_display import game_lines_comparison, odds_api_key_widget


@st.cache_data(ttl=1800)
def load_season_data(season: str):
    team_stats = get_team_season_stats(season)
    team_adv = get_team_advanced_stats(season)
    season_games = get_season_games(season)
    elo_ratings = compute_elo_ratings(season_games)
    all_teams = get_all_teams()
    return team_stats, team_adv, elo_ratings, all_teams


def render(season: str):
    st.title("🏠 Today's Games")
    st.caption(f"Generated lines based on {season} season data — {datetime.now().strftime('%B %d, %Y')}")

    # ── Odds API key (sidebar widget) ─────────────────────────────────────────
    api_key = odds_api_key_widget()

    with st.spinner("Loading today's schedule..."):
        try:
            today_games = get_todays_games()
        except Exception as e:
            st.error(f"Could not load today's schedule: {e}")
            today_games = pd.DataFrame()

    with st.spinner("Loading season stats & Elo ratings..."):
        try:
            team_stats, team_adv, elo_ratings, all_teams = load_season_data(season)
        except Exception as e:
            st.error(f"Could not load season data: {e}")
            return

    # ── Fetch sportsbook odds ─────────────────────────────────────────────────
    market_lines: dict = {}
    if api_key:
        with st.spinner("Fetching sportsbook odds..."):
            raw_odds = get_game_odds(api_key)
            if raw_odds and isinstance(raw_odds[0], dict) and "error" in raw_odds[0]:
                st.warning(f"Odds API: {raw_odds[0]['error']}")
            else:
                market_lines = parse_game_odds(raw_odds)
                if market_lines:
                    st.success(f"Live odds loaded from sportsbook for {len(market_lines)} games")

    if today_games.empty:
        st.warning("No games scheduled today.")
        st.info("Head to **Game Analyzer** to manually analyze any matchup.")
        return

    st.success(f"**{len(today_games)}** games today")
    st.markdown("---")

    team_lookup = {row["id"]: row for _, row in all_teams.iterrows()}

    for _, game in today_games.iterrows():
        home_id = int(game.get("HOME_TEAM_ID", 0))
        away_id = int(game.get("VISITOR_TEAM_ID", 0))
        if home_id == 0 or away_id == 0:
            continue

        home_team = team_lookup.get(home_id, {})
        away_team = team_lookup.get(away_id, {})
        home_name = home_team.get("full_name", f"Team {home_id}")
        away_name = away_team.get("full_name", f"Team {away_id}")

        # ── Model lines ───────────────────────────────────────────────────────
        home_elo = elo_ratings.get(home_id, 1500)
        away_elo = elo_ratings.get(away_id, 1500)
        home_wp, away_wp = win_probability(home_elo, away_elo)
        home_ml = prob_to_moneyline(home_wp)
        away_ml = prob_to_moneyline(away_wp)

        try:
            home_log = get_team_game_log(home_id, season, last_n=10)
            away_log = get_team_game_log(away_id, season, last_n=10)
            spread_result = expected_margin(home_id, away_id, team_stats, team_adv, home_log, away_log)
        except Exception:
            spread_result = {"spread": 0, "spread_line": 0, "total": 230}

        spread_val = spread_result["spread_line"]
        total = spread_result["total"]

        # ── Sportsbook lines ──────────────────────────────────────────────────
        book_lines = find_game_odds(market_lines, home_name, away_name) if market_lines else None

        # ── Game card ─────────────────────────────────────────────────────────
        game_time = game.get("GAME_STATUS_TEXT", "TBD")
        st.subheader(f"{away_name}  @  {home_name}  —  {game_time}")

        # Win probability bar
        bar_col1, bar_col2 = st.columns([home_wp, away_wp])
        with bar_col1:
            st.markdown(
                f"<div style='background:#2ecc71;border-radius:6px 0 0 6px;padding:4px 8px;"
                f"text-align:center;font-weight:700'>{home_name}: {home_wp*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
        with bar_col2:
            st.markdown(
                f"<div style='background:#e74c3c;border-radius:0 6px 6px 0;padding:4px 8px;"
                f"text-align:center;font-weight:700'>{away_name}: {away_wp*100:.1f}%</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-side lines table
        game_lines_comparison(
            home_name=home_name,
            away_name=away_name,
            our_home_ml=home_ml,
            our_away_ml=away_ml,
            our_spread=spread_val,
            our_total=total,
            book_lines=book_lines,
        )

        st.markdown("---")
