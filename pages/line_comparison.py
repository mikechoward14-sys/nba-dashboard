"""
Line Comparison page — compare our generated lines against market odds
from The Odds API (optional, requires free API key) or manual input.
"""
import streamlit as st
import pandas as pd
import requests
import os

from data.fetcher import get_all_teams, get_team_season_stats, get_team_advanced_stats, get_season_games, get_team_game_log
from models.elo import compute_elo_ratings, win_probability, prob_to_moneyline
from models.spread import expected_margin
from utils.formatting import fmt_moneyline, fmt_spread, spread_display

ODDS_API_KEY_ENV = "ODDS_API_KEY"
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


@st.cache_data(ttl=3600)
def load_season_data(season: str):
    team_stats = get_team_season_stats(season)
    team_adv = get_team_advanced_stats(season)
    season_games = get_season_games(season)
    elo_ratings = compute_elo_ratings(season_games)
    return team_stats, team_adv, elo_ratings


def fetch_market_odds(api_key: str) -> list[dict]:
    """Fetch current NBA odds from The Odds API."""
    try:
        resp = requests.get(
            ODDS_API_BASE,
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return []


def parse_market_lines(odds_data: list[dict]) -> pd.DataFrame:
    """Parse Odds API response into a flat DataFrame."""
    rows = []
    for game in odds_data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        for book in game.get("bookmakers", []):
            bk = book.get("key", "")
            for market in book.get("markets", []):
                mkt = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": bk,
                        "market": mkt,
                        "team": outcome.get("name", ""),
                        "price": outcome.get("price", 0),
                        "point": outcome.get("point", None),
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render(season: str):
    st.title("📈 Line Comparison")
    st.caption("Compare your generated lines against sportsbook market lines.")

    # ── API key input ─────────────────────────────────────────────────────────
    st.markdown("### Market Odds Source")
    api_tab, manual_tab = st.tabs(["The Odds API (Free Key)", "Manual Input"])

    market_lines = pd.DataFrame()

    with api_tab:
        st.markdown(
            "Get a free API key at [the-odds-api.com](https://the-odds-api.com) — 500 requests/month free."
        )
        api_key = st.text_input(
            "Odds API Key",
            value=os.getenv(ODDS_API_KEY_ENV, ""),
            type="password",
            placeholder="Enter your free API key...",
        )
        if api_key and st.button("Fetch Market Odds"):
            with st.spinner("Fetching live odds..."):
                raw = fetch_market_odds(api_key)
                if raw:
                    market_lines = parse_market_lines(raw)
                    st.success(f"Fetched odds for {len(raw)} games.")
                else:
                    st.error("Could not fetch odds. Check your API key.")

    with manual_tab:
        st.markdown("Enter market lines manually to compare against our model.")
        manual_home = st.text_input("Home Team")
        manual_away = st.text_input("Away Team")
        m1, m2, m3 = st.columns(3)
        with m1:
            mkt_home_ml = st.number_input("Market Home ML", value=-110)
        with m2:
            mkt_away_ml = st.number_input("Market Away ML", value=-110)
        with m3:
            mkt_spread = st.number_input("Market Home Spread", value=-3.0, step=0.5)
        mkt_total = st.number_input("Market Total", value=220.0, step=0.5)

        if manual_home and manual_away:
            market_lines = pd.DataFrame([{
                "home_team": manual_home,
                "away_team": manual_away,
                "mkt_home_ml": mkt_home_ml,
                "mkt_away_ml": mkt_away_ml,
                "mkt_spread": mkt_spread,
                "mkt_total": mkt_total,
                "manual": True,
            }])

    st.markdown("---")

    # ── Load our model lines ──────────────────────────────────────────────────
    with st.spinner("Loading model data..."):
        try:
            team_stats, team_adv, elo_ratings = load_season_data(season)
        except Exception as e:
            st.error(f"Could not load model data: {e}")
            return

    all_teams = get_all_teams()
    team_name_to_id = {row["full_name"]: row["id"] for _, row in all_teams.iterrows()}
    team_abbr_to_id = {row["abbreviation"]: row["id"] for _, row in all_teams.iterrows()}

    # ── Manual comparison ─────────────────────────────────────────────────────
    if not market_lines.empty and "manual" in market_lines.columns:
        row = market_lines.iloc[0]
        home_name = row["home_team"]
        away_name = row["away_team"]

        home_id = team_name_to_id.get(home_name) or next(
            (v for k, v in team_name_to_id.items() if home_name.lower() in k.lower()), None
        )
        away_id = team_name_to_id.get(away_name) or next(
            (v for k, v in team_name_to_id.items() if away_name.lower() in k.lower()), None
        )

        if home_id and away_id:
            home_elo = elo_ratings.get(int(home_id), 1500)
            away_elo = elo_ratings.get(int(away_id), 1500)
            home_wp, away_wp = win_probability(home_elo, away_elo)
            our_home_ml = prob_to_moneyline(home_wp)
            our_away_ml = prob_to_moneyline(away_wp)

            try:
                home_log = get_team_game_log(int(home_id), season, last_n=10)
                away_log = get_team_game_log(int(away_id), season, last_n=10)
                spread_result = expected_margin(int(home_id), int(away_id), team_stats, team_adv, home_log, away_log)
            except Exception:
                spread_result = {"spread_line": 0, "total": 220}

            our_spread = spread_result["spread_line"]
            our_total = spread_result["total"]

            st.subheader(f"{away_name} @ {home_name}")
            _show_comparison(
                our_home_ml, int(row["mkt_home_ml"]),
                our_away_ml, int(row["mkt_away_ml"]),
                our_spread, float(row["mkt_spread"]),
                our_total, float(row["mkt_total"]),
                home_name, away_name,
            )
        else:
            st.warning("Could not match team names to NBA database. Try full team names (e.g. 'Boston Celtics').")

    # ── API-based comparison ──────────────────────────────────────────────────
    elif not market_lines.empty and "manual" not in market_lines.columns:
        games = market_lines["home_team"].unique()
        for home_name in games:
            game_rows = market_lines[market_lines["home_team"] == home_name]
            away_name = game_rows["away_team"].iloc[0]

            home_id = next((v for k, v in team_name_to_id.items() if home_name in k), None)
            away_id = next((v for k, v in team_name_to_id.items() if away_name in k), None)
            if not home_id or not away_id:
                continue

            home_elo = elo_ratings.get(int(home_id), 1500)
            away_elo = elo_ratings.get(int(away_id), 1500)
            home_wp, away_wp = win_probability(home_elo, away_elo)
            our_home_ml = prob_to_moneyline(home_wp)
            our_away_ml = prob_to_moneyline(away_wp)

            # Get average market ML
            h2h_rows = game_rows[game_rows["market"] == "h2h"]
            home_mls = h2h_rows[h2h_rows["team"] == home_name]["price"].tolist()
            away_mls = h2h_rows[h2h_rows["team"] == away_name]["price"].tolist()
            mkt_home_ml = int(sum(home_mls) / len(home_mls)) if home_mls else 0
            mkt_away_ml = int(sum(away_mls) / len(away_mls)) if away_mls else 0

            spread_rows = game_rows[(game_rows["market"] == "spreads") & (game_rows["team"] == home_name)]
            mkt_spread = float(spread_rows["point"].mean()) if not spread_rows.empty else 0

            total_rows = game_rows[game_rows["market"] == "totals"]
            mkt_total = float(total_rows["point"].mean()) if not total_rows.empty else 220

            try:
                home_log = get_team_game_log(int(home_id), season, last_n=10)
                away_log = get_team_game_log(int(away_id), season, last_n=10)
                spread_result = expected_margin(int(home_id), int(away_id), team_stats, team_adv, home_log, away_log)
            except Exception:
                spread_result = {"spread_line": 0, "total": 220}

            our_spread = spread_result["spread_line"]
            our_total = spread_result["total"]

            with st.container():
                st.subheader(f"{away_name} @ {home_name}")
                _show_comparison(
                    our_home_ml, mkt_home_ml,
                    our_away_ml, mkt_away_ml,
                    our_spread, mkt_spread,
                    our_total, mkt_total,
                    home_name, away_name,
                )
            st.markdown("---")

    else:
        st.info("Enter market lines manually or connect The Odds API to compare against our model.")

        # Show our picks for common matchups as a demo
        st.markdown("### Model Lines Preview (select matchup in Game Analyzer)")
        st.markdown("Use the **Game Analyzer** tab to generate lines for any specific matchup.")


def _show_comparison(
    our_home_ml, mkt_home_ml,
    our_away_ml, mkt_away_ml,
    our_spread, mkt_spread,
    our_total, mkt_total,
    home_name, away_name,
):
    """Render a side-by-side comparison card."""
    col_label, col_ours, col_market, col_edge = st.columns([2, 2, 2, 2])
    col_label.markdown("**Line**")
    col_ours.markdown("**Our Line**")
    col_market.markdown("**Market**")
    col_edge.markdown("**Edge**")

    rows = [
        ("Home ML", fmt_moneyline(our_home_ml), fmt_moneyline(mkt_home_ml)),
        ("Away ML", fmt_moneyline(our_away_ml), fmt_moneyline(mkt_away_ml)),
        ("Spread (Home)", fmt_spread(-our_spread), fmt_spread(-mkt_spread)),
        ("Total O/U", f"{our_total:.1f}", f"{mkt_total:.1f}"),
    ]

    for label, ours, market in rows:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        c1.write(label)
        c2.write(f"**{ours}**")
        c3.write(market)

        # Simple edge calc for ML
        if "ML" in label:
            try:
                our_val = int(ours.replace("+", ""))
                mkt_val = int(market.replace("+", ""))
                diff = our_val - mkt_val
                if abs(diff) >= 5:
                    edge_str = f"{'🟢' if diff > 0 else '🔴'} {diff:+d}"
                else:
                    edge_str = "≈ Fair"
                c4.write(edge_str)
            except Exception:
                c4.write("—")
        elif "Spread" in label:
            try:
                our_val = float(ours)
                mkt_val = float(market)
                diff = our_val - mkt_val
                if abs(diff) >= 0.5:
                    c4.write(f"{'🟢' if diff > 0 else '🔴'} {diff:+.1f}")
                else:
                    c4.write("≈ Fair")
            except Exception:
                c4.write("—")
        elif "Total" in label:
            try:
                diff = float(ours) - float(market)
                if abs(diff) >= 1:
                    c4.write(f"{'⬆️' if diff > 0 else '⬇️'} {diff:+.1f}")
                else:
                    c4.write("≈ Fair")
            except Exception:
                c4.write("—")
        else:
            c4.write("—")
