"""
Player Props page — generate over/under lines for any player vs any opponent.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.fetcher import (
    get_all_active_players,
    get_all_teams,
    get_player_game_log,
    get_team_season_stats,
    get_team_advanced_stats,
)
from models.player_props import all_props_for_player, PROP_CATEGORIES
from utils.formatting import fmt_moneyline, hit_rate_label
from data.odds_fetcher import get_all_player_props, find_player_props
from utils.line_display import prop_line_comparison, prop_table_header, odds_api_key_widget


@st.cache_data(ttl=3600)
def load_player_list():
    return get_all_active_players()

@st.cache_data(ttl=3600)
def load_team_data(season: str):
    stats = get_team_season_stats(season)
    adv = get_team_advanced_stats(season)
    return stats, adv


def render(season: str):
    st.title("👤 Player Props")
    st.caption("Generate over/under lines and probabilities for any NBA player.")

    api_key = odds_api_key_widget()

    with st.spinner("Loading player list..."):
        players_df = load_player_list()
    with st.spinner("Loading team stats..."):
        team_stats, team_adv = load_team_data(season)

    # Fetch all sportsbook player props upfront
    sportsbook_props: dict = {}
    if api_key:
        with st.spinner("Fetching sportsbook player props..."):
            sportsbook_props = get_all_player_props(api_key)

    all_teams = get_all_teams()

    # ── Player selection ──────────────────────────────────────────────────────
    col_player, col_opp = st.columns(2)

    with col_player:
        player_names = sorted(players_df["full_name"].tolist())
        player_name = st.selectbox("Select Player", player_names,
                                    index=player_names.index("LeBron James") if "LeBron James" in player_names else 0)
        player_row = players_df[players_df["full_name"] == player_name].iloc[0]
        player_id = int(player_row["id"])

    with col_opp:
        team_names = sorted(all_teams["full_name"].tolist())
        opp_name = st.selectbox("Opponent Team (for matchup adjustment)", ["None (season avg)"] + team_names)

    col_cats, col_window = st.columns(2)
    with col_cats:
        selected_cats = st.multiselect(
            "Stat Categories",
            list(PROP_CATEGORIES.keys()),
            default=["PTS", "REB", "AST", "FG3M"],
            format_func=lambda x: PROP_CATEGORIES[x],
        )
    with col_window:
        last_n = st.slider("Games for recent form", min_value=5, max_value=30, value=10)

    if not selected_cats:
        st.info("Select at least one stat category.")
        return

    # ── Load game log ─────────────────────────────────────────────────────────
    with st.spinner(f"Loading {player_name}'s game log..."):
        try:
            game_log = get_player_game_log(player_id, season, last_n=30)
        except Exception as e:
            st.error(f"Could not load game log: {e}")
            return

    if game_log.empty:
        st.warning(f"No game log found for {player_name} in {season}.")
        return

    # ── Opponent stats for matchup adjustment ────────────────────────────────
    opp_stats_row = None
    game_pace = 100.0
    league_avg_pace = 100.0

    if opp_name != "None (season avg)" and not team_stats.empty:
        opp_team_row = all_teams[all_teams["full_name"] == opp_name]
        if not opp_team_row.empty:
            opp_team_id = int(opp_team_row.iloc[0]["id"])
            opp_mask = team_stats["TEAM_ID"] == opp_team_id
            if opp_mask.any():
                opp_stats_row = team_stats[opp_mask].iloc[0]
            # Pace
            if not team_adv.empty:
                opp_adv_mask = team_adv["TEAM_ID"] == opp_team_id
                if opp_adv_mask.any():
                    game_pace = float(team_adv[opp_adv_mask].iloc[0].get("PACE", 100))
                    league_avg_pace = float(team_adv["PACE"].mean()) if "PACE" in team_adv else 100.0

    # ── Generate props ────────────────────────────────────────────────────────
    props = all_props_for_player(
        game_log,
        opponent_stats=opp_stats_row,
        game_pace=game_pace,
        league_avg_pace=league_avg_pace,
        categories=selected_cats,
    )

    if not props:
        st.warning("Could not generate props — missing stat columns in game log.")
        return

    # Match sportsbook props for this player
    player_book_props = find_player_props(sportsbook_props, player_name) if sportsbook_props else {}

    # ── Side-by-side props table ──────────────────────────────────────────────
    st.markdown(f"### {player_name} — Our Lines vs Sportsbook")
    if opp_name != "None (season avg)":
        st.caption(f"Matchup-adjusted vs {opp_name}")

    prop_table_header()
    for prop in props:
        book_data = player_book_props.get(prop["stat"])
        prop_line_comparison(
            stat_label=prop["stat_label"],
            our_line=prop["line"],
            our_proj=prop["projection"],
            our_over_ml=prop.get("ml_over", 0),
            our_under_ml=prop.get("ml_under", 0),
            our_hit_rate=prop["hit_rate_over"],
            book_data=book_data,
        )

    # ── Per-stat detail charts ────────────────────────────────────────────────
    st.markdown("### Game Log Charts")
    for prop in props:
        bk = player_book_props.get(prop["stat"])
        bk_line = bk["line"] if bk else None
        expander_label = f"{prop['stat_label']} — Ours: {prop['line']}"
        if bk_line:
            expander_label += f"  |  Book: {bk_line:.1f}"
        with st.expander(expander_label):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Our Line", prop["line"])
            c2.metric("Book Line", f"{bk_line:.1f}" if bk_line else "—")
            c3.metric("Projection", prop["projection"])
            c4.metric("Hist Over %", hit_rate_label(prop["hit_rate_over"]))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Season Avg", prop["season_avg"])
            c6.metric(f"Last {last_n} Avg", prop["recent_avg"])
            c7.metric("Our Over ML", fmt_moneyline(prop.get("ml_over", 0)))
            c8.metric("Book Over ML", fmt_moneyline(bk["over_ml"]) if bk and bk.get("over_ml") else "—")

            stat_col = prop["stat"]
            if stat_col in game_log.columns:
                series = pd.to_numeric(game_log[stat_col], errors="coerce").dropna()
                display_line = bk_line if bk_line else prop["line"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(1, len(series) + 1)),
                    y=series.values,
                    marker_color=["#2ecc71" if v > display_line else "#e74c3c" for v in series.values],
                    name=prop["stat_label"],
                ))
                fig.add_hline(y=prop["line"], line_dash="dash", line_color="gold",
                              annotation_text=f"Our: {prop['line']}")
                if bk_line and bk_line != prop["line"]:
                    fig.add_hline(y=bk_line, line_dash="dot", line_color="#3498db",
                                  annotation_text=f"Book: {bk_line:.1f}")
                fig.update_layout(
                    title=f"Last {len(series)} Games — {prop['stat_label']}",
                    xaxis_title="Game (most recent first)",
                    yaxis_title=prop["stat_label"],
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=280,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
