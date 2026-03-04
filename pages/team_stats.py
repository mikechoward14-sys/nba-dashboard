"""
Team Stats page — league-wide sortable table of team stats with ratings.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from data.fetcher import (
    get_team_season_stats,
    get_team_advanced_stats,
    get_season_games,
    get_all_teams,
)
from models.elo import compute_elo_ratings


@st.cache_data(ttl=3600)
def load_all_stats(season: str):
    basic = get_team_season_stats(season)
    adv = get_team_advanced_stats(season)
    season_games = get_season_games(season)
    elo = compute_elo_ratings(season_games)
    return basic, adv, elo


def render(season: str):
    st.title("📊 Team Stats")
    st.caption(f"League-wide team statistics — {season}")

    with st.spinner("Loading team stats..."):
        try:
            basic, adv, elo_ratings = load_all_stats(season)
        except Exception as e:
            st.error(f"Could not load stats: {e}")
            return

    if basic.empty:
        st.warning("No stats data available.")
        return

    # ── Merge basic + advanced ────────────────────────────────────────────────
    merge_cols = ["TEAM_ID", "TEAM_NAME"]
    if not adv.empty and "TEAM_ID" in adv.columns:
        adv_cols = ["TEAM_ID"] + [c for c in adv.columns if c not in basic.columns]
        merged = basic.merge(adv[adv_cols], on="TEAM_ID", how="left")
    else:
        merged = basic.copy()

    # Add Elo
    if "TEAM_ID" in merged.columns:
        merged["ELO"] = merged["TEAM_ID"].map(elo_ratings).round(1)

    # ── Filter / sort controls ────────────────────────────────────────────────
    col_sort, col_dir = st.columns(2)
    numeric_cols = merged.select_dtypes(include="number").columns.tolist()
    default_sort = "ELO" if "ELO" in numeric_cols else numeric_cols[0] if numeric_cols else None

    with col_sort:
        sort_col = st.selectbox("Sort by", numeric_cols, index=numeric_cols.index(default_sort) if default_sort in numeric_cols else 0)
    with col_dir:
        ascending = st.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Ascending"

    # Display columns
    display_cols = ["TEAM_NAME"]
    for c in ["W", "L", "WIN_PCT", "PTS", "OPP_PTS", "PLUS_MINUS", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "ELO"]:
        if c in merged.columns:
            display_cols.append(c)

    display_df = merged[display_cols].sort_values(sort_col, ascending=ascending) if sort_col else merged[display_cols]
    display_df = display_df.rename(columns={"TEAM_NAME": "Team"})

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Scatter: OffRtg vs DefRtg ─────────────────────────────────────────────
    st.markdown("### Offensive vs Defensive Rating")
    if "OFF_RATING" in merged.columns and "DEF_RATING" in merged.columns:
        fig = px.scatter(
            merged,
            x="DEF_RATING",
            y="OFF_RATING",
            text="TEAM_NAME",
            color="ELO" if "ELO" in merged.columns else None,
            color_continuous_scale="RdYlGn",
            title="Four Quadrants: OffRtg vs DefRtg (lower DefRtg = better defense)",
            labels={"DEF_RATING": "Defensive Rating (lower = better)", "OFF_RATING": "Offensive Rating"},
        )
        fig.update_traces(textposition="top center", marker_size=12)
        # Quadrant lines
        avg_off = merged["OFF_RATING"].mean()
        avg_def = merged["DEF_RATING"].mean()
        fig.add_vline(x=avg_def, line_dash="dash", line_color="gray")
        fig.add_hline(y=avg_off, line_dash="dash", line_color="gray")
        fig.update_layout(height=550, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Elo rankings ─────────────────────────────────────────────────────────
    if "ELO" in merged.columns:
        st.markdown("### Elo Power Rankings")
        elo_df = merged[["TEAM_NAME", "ELO"]].dropna().sort_values("ELO", ascending=True)
        fig_elo = px.bar(
            elo_df,
            x="ELO",
            y="TEAM_NAME",
            orientation="h",
            color="ELO",
            color_continuous_scale="RdYlGn",
            title="Team Elo Ratings",
        )
        fig_elo.update_layout(height=700, paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
        st.plotly_chart(fig_elo, use_container_width=True)
