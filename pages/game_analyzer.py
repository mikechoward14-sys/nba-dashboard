"""
Game Analyzer — pick any two teams and get a full matchup breakdown.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data.fetcher import (
    get_all_teams,
    get_team_season_stats,
    get_team_advanced_stats,
    get_team_game_log,
    get_season_games,
    get_head_to_head,
)
from models.elo import compute_elo_ratings, win_probability, prob_to_moneyline
from models.spread import expected_margin
from utils.formatting import fmt_moneyline, fmt_spread, spread_display
from data.odds_fetcher import get_game_odds, parse_game_odds, find_game_odds
from utils.line_display import game_lines_comparison, odds_api_key_widget


@st.cache_data(ttl=3600)
def load_season_data(season: str):
    team_stats = get_team_season_stats(season)
    team_adv = get_team_advanced_stats(season)
    season_games = get_season_games(season)
    elo_ratings = compute_elo_ratings(season_games)
    return team_stats, team_adv, elo_ratings


def render(season: str):
    st.title("🔮 Game Analyzer")
    st.caption("Select any two teams for a detailed matchup breakdown and projected lines.")

    api_key = odds_api_key_widget()

    all_teams = get_all_teams()
    team_names = sorted(all_teams["full_name"].tolist())
    team_id_map = {row["full_name"]: row["id"] for _, row in all_teams.iterrows()}

    col_home, col_away = st.columns(2)
    with col_home:
        home_name = st.selectbox("🏠 Home Team", team_names, index=team_names.index("Boston Celtics") if "Boston Celtics" in team_names else 0)
    with col_away:
        default_away = "Miami Heat" if "Miami Heat" in team_names else team_names[1]
        away_name = st.selectbox("✈️ Away Team", [t for t in team_names if t != home_name], index=0)

    if home_name == away_name:
        st.warning("Select two different teams.")
        return

    home_id = int(team_id_map[home_name])
    away_id = int(team_id_map[away_name])

    with st.spinner("Loading season data..."):
        try:
            team_stats, team_adv, elo_ratings = load_season_data(season)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    with st.spinner("Loading game logs..."):
        try:
            home_log = get_team_game_log(home_id, season, last_n=20)
            away_log = get_team_game_log(away_id, season, last_n=20)
        except Exception as e:
            st.error(f"Failed to load game logs: {e}")
            home_log = pd.DataFrame()
            away_log = pd.DataFrame()

    # ── Win probability ───────────────────────────────────────────────────────
    home_elo = elo_ratings.get(home_id, 1500)
    away_elo = elo_ratings.get(away_id, 1500)
    home_wp, away_wp = win_probability(home_elo, away_elo)
    home_ml = prob_to_moneyline(home_wp)
    away_ml = prob_to_moneyline(away_wp)

    # ── Spread ────────────────────────────────────────────────────────────────
    try:
        spread_result = expected_margin(
            home_id, away_id, team_stats, team_adv, home_log, away_log
        )
    except Exception as e:
        st.warning(f"Spread model error: {e}")
        spread_result = {"spread": 0, "spread_line": 0, "total": 220,
                         "home_implied_pts": 110, "away_implied_pts": 110, "details": {}}

    spread_val = spread_result["spread_line"]
    home_spread_str, away_spread_str = spread_display(spread_val)

    # ── Fetch sportsbook lines ────────────────────────────────────────────────
    book_lines = None
    if api_key:
        with st.spinner("Fetching sportsbook odds..."):
            raw_odds = get_game_odds(api_key)
            if raw_odds and not (isinstance(raw_odds[0], dict) and "error" in raw_odds[0]):
                market_lines = parse_game_odds(raw_odds)
                book_lines = find_game_odds(market_lines, home_name, away_name)

    # ── Header summary ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"## {away_name}  **@**  {home_name}")

    # Win prob metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Home Win Prob", f"{home_wp*100:.1f}%")
    c2.metric("Away Win Prob", f"{away_wp*100:.1f}%")
    c3.metric("Elo (Home)", f"{home_elo:.0f}")
    c4.metric("Elo (Away)", f"{away_elo:.0f}")
    c5.metric("Implied Total", f"{spread_result['total']:.1f}")

    st.markdown("### Lines: Our Model vs Sportsbook")
    game_lines_comparison(
        home_name=home_name,
        away_name=away_name,
        our_home_ml=home_ml,
        our_away_ml=away_ml,
        our_spread=spread_val,
        our_total=spread_result["total"],
        book_lines=book_lines,
    )

    # ── Win probability gauge ─────────────────────────────────────────────────
    st.markdown("### Win Probability")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=home_wp * 100,
        title={"text": f"{home_name} Win %"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2ecc71"},
            "steps": [
                {"range": [0, 40], "color": "#e74c3c"},
                {"range": [40, 60], "color": "#f39c12"},
                {"range": [60, 100], "color": "#27ae60"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "value": 50},
        },
    ))
    fig_gauge.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Rating comparison ─────────────────────────────────────────────────────
    st.markdown("### Team Ratings Comparison")
    details = spread_result.get("details", {})
    if details:
        categories = ["Off Rating", "Def Rating", "Pace", "Elo"]
        home_vals = [
            details.get("home_off_rtg", 0),
            # Invert def rating so higher = better
            200 - details.get("home_def_rtg", 0),
            details.get("home_pace", 0),
            home_elo / 20,  # scale to same range
        ]
        away_vals = [
            details.get("away_off_rtg", 0),
            200 - details.get("away_def_rtg", 0),
            details.get("away_pace", 0),
            away_elo / 20,
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=home_vals, theta=categories, fill="toself",
            name=home_name, line_color="#2ecc71",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=away_vals, theta=categories, fill="toself",
            name=away_name, line_color="#e74c3c",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False)),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=350,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Recent form chart ─────────────────────────────────────────────────────
    st.markdown("### Recent Form (Last 20 Games)")
    form_col1, form_col2 = st.columns(2)

    for col, log, name, color in [
        (form_col1, home_log, home_name, "#2ecc71"),
        (form_col2, away_log, away_name, "#e74c3c"),
    ]:
        with col:
            st.subheader(name)
            if not log.empty and "PLUS_MINUS" in log.columns:
                log_sorted = log.copy()
                log_sorted["GAME_NUM"] = range(1, len(log_sorted) + 1)
                log_sorted["PLUS_MINUS"] = pd.to_numeric(log_sorted["PLUS_MINUS"], errors="coerce")
                log_sorted["COLOR"] = log_sorted["PLUS_MINUS"].apply(
                    lambda x: "#2ecc71" if x >= 0 else "#e74c3c"
                )
                fig_bar = go.Figure(go.Bar(
                    x=log_sorted["GAME_NUM"],
                    y=log_sorted["PLUS_MINUS"],
                    marker_color=log_sorted["COLOR"],
                    text=log_sorted.get("MATCHUP", ""),
                    hovertext=log_sorted.get("MATCHUP", ""),
                ))
                fig_bar.update_layout(
                    title="Point Differential",
                    xaxis_title="Game (most recent first)",
                    yaxis_title="Point Diff",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=280,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                wins = (log_sorted.get("WL", pd.Series()) == "W").sum()
                losses = len(log_sorted) - wins
                avg_margin = log_sorted["PLUS_MARGIN"].mean() if "PLUS_MARGIN" in log_sorted else log_sorted["PLUS_MINUS"].mean()
                st.metric("Record (last 20)", f"{wins}-{losses}")
                st.metric("Avg Margin", f"{avg_margin:+.1f}")
            else:
                st.info("No game log data available.")

    # ── Head-to-head ──────────────────────────────────────────────────────────
    st.markdown("### Head-to-Head History")
    with st.spinner("Loading H2H..."):
        try:
            h2h = get_head_to_head(home_id, away_id, last_n_seasons=3)
        except Exception:
            h2h = pd.DataFrame()

    if not h2h.empty:
        display_cols = [c for c in ["GAME_DATE", "MATCHUP", "WL", "PTS", "PLUS_MINUS"] if c in h2h.columns]
        st.dataframe(h2h[display_cols].head(15), use_container_width=True)
    else:
        st.info("No head-to-head data available.")

    # ── Model breakdown ───────────────────────────────────────────────────────
    with st.expander("📊 Model Details"):
        st.markdown("**Elo Ratings**")
        st.json({"home_elo": round(home_elo, 1), "away_elo": round(away_elo, 1),
                 "home_win_prob": home_wp, "away_win_prob": away_wp})
        st.markdown("**Spread Model Inputs**")
        st.json(details)
        st.markdown("**Projected Scores**")
        st.json({
            "home_implied_pts": spread_result.get("home_implied_pts"),
            "away_implied_pts": spread_result.get("away_implied_pts"),
            "total": spread_result.get("total"),
        })
