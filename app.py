"""
NBA Betting Lines Dashboard — Entry Point
Run with: streamlit run app.py
"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

st.set_page_config(
    page_title="NBA Lines Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .team-header { font-size: 1.1rem; font-weight: 700; }
    .line-value  { font-size: 1.6rem; font-weight: 800; }
    .green { color: #2ecc71; }
    .red   { color: #e74c3c; }
    .gold  { color: #f1c40f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("🏀 NBA Lines Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Today's Games",
        "🔮 Game Analyzer",
        "👤 Player Props",
        "💰 Value Bets & Parlays",
        "📉 Model Performance",
        "📊 Team Stats",
        "📈 Line Comparison",
    ],
)

st.sidebar.markdown("---")
season = st.sidebar.selectbox(
    "Season",
    ["2025-26", "2024-25", "2023-24", "2022-23"],
    index=0,
)
st.sidebar.caption("Data via NBA.com (nba_api)")

# ── Route pages ───────────────────────────────────────────────────────────────
if page == "🏠 Today's Games":
    from pages.todays_games import render
    render(season)
elif page == "🔮 Game Analyzer":
    from pages.game_analyzer import render
    render(season)
elif page == "👤 Player Props":
    from pages.player_props import render
    render(season)
elif page == "💰 Value Bets & Parlays":
    from pages.value_bets import render
    render(season)
elif page == "📉 Model Performance":
    from pages.model_performance import render
    render(season)
elif page == "📊 Team Stats":
    from pages.team_stats import render
    render(season)
elif page == "📈 Line Comparison":
    from pages.line_comparison import render
    render(season)
