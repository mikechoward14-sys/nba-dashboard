"""
Value Bets & Parlays page.

Scans all of today's games + player props, finds edges between our model
and the sportsbook, ranks them by expected value, and suggests parlays.

Edge = our implied prob is meaningfully higher than book's implied prob.
Expected Value = (our_prob * (payout + 1)) - 1
"""
import streamlit as st
import pandas as pd
import itertools

from data.fetcher import (
    get_todays_games, get_all_teams, get_team_season_stats,
    get_team_advanced_stats, get_team_game_log, get_season_games,
    get_all_active_players, get_player_game_log, get_team_advanced_stats,
)
from data.odds_fetcher import (
    get_game_odds, parse_game_odds, find_game_odds,
    get_all_player_props, find_player_props,
)
from models.elo import compute_elo_ratings, win_probability, prob_to_moneyline
from models.spread import expected_margin
from models.player_props import all_props_for_player, PROP_CATEGORIES
from utils.line_display import odds_api_key_widget
from utils.formatting import fmt_moneyline

MIN_EDGE_PCT = 3.0      # minimum edge % to flag as value
MIN_EV = 0.02           # minimum expected value (2 cents per $1 bet)
MAX_PARLAY_LEGS = 4


# ── Math helpers ──────────────────────────────────────────────────────────────

def ml_to_decimal(ml: int) -> float:
    if ml > 0:
        return ml / 100 + 1
    return 100 / abs(ml) + 1

def ml_to_prob(ml: int) -> float:
    if ml > 0:
        return 100 / (ml + 100)
    return abs(ml) / (abs(ml) + 100)

def prob_to_ml(prob: float) -> int:
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return -round((prob / (1 - prob)) * 100)
    return round(((1 - prob) / prob) * 100)

def expected_value(our_prob: float, book_ml: int) -> float:
    """EV per $1 wagered."""
    decimal_odds = ml_to_decimal(book_ml)
    return our_prob * decimal_odds - 1

def edge_pct(our_prob: float, book_ml: int) -> float:
    book_prob = ml_to_prob(book_ml)
    return (our_prob - book_prob) * 100

def parlay_ml(legs: list[int]) -> int:
    """Combine multiple moneylines into a parlay payout."""
    combined = 1.0
    for ml in legs:
        combined *= ml_to_decimal(ml)
    # Convert back to American
    if combined >= 2.0:
        return round((combined - 1) * 100)
    return -round(100 / (combined - 1))

def confidence_stars(ev: float) -> str:
    if ev >= 0.12:
        return "⭐⭐⭐⭐⭐"
    elif ev >= 0.08:
        return "⭐⭐⭐⭐"
    elif ev >= 0.05:
        return "⭐⭐⭐"
    elif ev >= 0.03:
        return "⭐⭐"
    return "⭐"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_game_model_data(season: str):
    team_stats = get_team_season_stats(season)
    team_adv = get_team_advanced_stats(season)
    season_games = get_season_games(season)
    elo_ratings = compute_elo_ratings(season_games)
    all_teams = get_all_teams()
    return team_stats, team_adv, elo_ratings, all_teams


# ── Value bet scanner ─────────────────────────────────────────────────────────

def scan_game_bets(today_games, team_stats, team_adv, elo_ratings, team_lookup, market_lines, season) -> list[dict]:
    bets = []
    for _, game in today_games.iterrows():
        home_id = int(game.get("HOME_TEAM_ID", 0))
        away_id = int(game.get("VISITOR_TEAM_ID", 0))
        if home_id == 0 or away_id == 0:
            continue

        home_name = team_lookup.get(home_id, {}).get("full_name", str(home_id))
        away_name = team_lookup.get(away_id, {}).get("full_name", str(away_id))
        game_label = f"{away_name} @ {home_name}"

        # Model lines
        home_elo = elo_ratings.get(home_id, 1500)
        away_elo = elo_ratings.get(away_id, 1500)
        home_wp, away_wp = win_probability(home_elo, away_elo)

        try:
            home_log = get_team_game_log(home_id, season, last_n=10)
            away_log = get_team_game_log(away_id, season, last_n=10)
            spread_res = expected_margin(home_id, away_id, team_stats, team_adv, home_log, away_log)
        except Exception:
            spread_res = {"spread_line": 0, "total": 230}

        book = find_game_odds(market_lines, home_name, away_name) if market_lines else None
        if not book:
            continue

        checks = [
            ("Moneyline", home_name, home_wp, book.get("home_ml"), "game"),
            ("Moneyline", away_name, away_wp, book.get("away_ml"), "game"),
        ]

        for bet_type, side, our_prob, book_ml, category in checks:
            if not book_ml:
                continue
            ev = expected_value(our_prob, book_ml)
            ep = edge_pct(our_prob, book_ml)
            if ep >= MIN_EDGE_PCT and ev >= MIN_EV:
                bets.append({
                    "game": game_label,
                    "bet": f"{bet_type}: {side}",
                    "category": category,
                    "our_prob": our_prob,
                    "our_ml": prob_to_ml(our_prob),
                    "book_ml": book_ml,
                    "edge_pct": round(ep, 2),
                    "ev": round(ev, 4),
                    "confidence": confidence_stars(ev),
                    "book": book.get("book", "Book"),
                })
    return bets


def scan_prop_bets(sportsbook_props: dict, season: str) -> list[dict]:
    """Scan all players with book props, generate our lines, find edges."""
    bets = []
    if not sportsbook_props:
        return bets

    players_df = get_all_active_players()
    player_name_map = {row["full_name"]: row["id"] for _, row in players_df.iterrows()}

    for player_name, stat_props in sportsbook_props.items():
        player_id = player_name_map.get(player_name)
        if not player_id:
            # Try last-name match
            last = player_name.split()[-1].lower()
            for pn, pid in player_name_map.items():
                if pn.split()[-1].lower() == last:
                    player_id = pid
                    break
        if not player_id:
            continue

        try:
            game_log = get_player_game_log(int(player_id), season, last_n=25)
        except Exception:
            continue
        if game_log.empty:
            continue

        our_props = all_props_for_player(game_log, categories=list(stat_props.keys()))

        for our_prop in our_props:
            stat = our_prop["stat"]
            book_data = stat_props.get(stat)
            if not book_data:
                continue

            bk_over_ml = book_data.get("over_ml")
            bk_under_ml = book_data.get("under_ml")
            bk_line = book_data.get("line")

            if not bk_over_ml or not bk_under_ml or not bk_line:
                continue

            # Check over
            over_prob = our_prop["over_prob"]
            ev_over = expected_value(over_prob, bk_over_ml)
            ep_over = edge_pct(over_prob, bk_over_ml)
            if ep_over >= MIN_EDGE_PCT and ev_over >= MIN_EV:
                bets.append({
                    "game": player_name,
                    "bet": f"{PROP_CATEGORIES.get(stat, stat)} Over {bk_line:.1f}",
                    "category": "prop",
                    "our_prob": over_prob,
                    "our_ml": our_prop.get("ml_over", 0),
                    "book_ml": bk_over_ml,
                    "edge_pct": round(ep_over, 2),
                    "ev": round(ev_over, 4),
                    "confidence": confidence_stars(ev_over),
                    "book": book_data.get("book", "Book"),
                    "hit_rate": our_prop["hit_rate_over"],
                })

            # Check under
            under_prob = our_prop["under_prob"]
            ev_under = expected_value(under_prob, bk_under_ml)
            ep_under = edge_pct(under_prob, bk_under_ml)
            if ep_under >= MIN_EDGE_PCT and ev_under >= MIN_EV:
                bets.append({
                    "game": player_name,
                    "bet": f"{PROP_CATEGORIES.get(stat, stat)} Under {bk_line:.1f}",
                    "category": "prop",
                    "our_prob": under_prob,
                    "our_ml": our_prop.get("ml_under", 0),
                    "book_ml": bk_under_ml,
                    "edge_pct": round(ep_under, 2),
                    "ev": round(ev_under, 4),
                    "confidence": confidence_stars(ev_under),
                    "book": book_data.get("book", "Book"),
                    "hit_rate": 1 - our_prop["hit_rate_over"],
                })
    return bets


def build_parlays(value_bets: list[dict], max_legs: int = MAX_PARLAY_LEGS) -> list[dict]:
    """Build top parlay combinations from value bets, ranked by combined EV."""
    if len(value_bets) < 2:
        return []

    # Sort by EV, take top candidates
    candidates = sorted(value_bets, key=lambda x: x["ev"], reverse=True)[:10]
    parlays = []

    for n_legs in range(2, min(max_legs + 1, len(candidates) + 1)):
        for combo in itertools.combinations(candidates, n_legs):
            legs_ml = [b["book_ml"] for b in combo]
            combined_ml = parlay_ml(legs_ml)
            combined_prob = 1.0
            for b in combo:
                combined_prob *= b["our_prob"]
            ev = expected_value(combined_prob, combined_ml)
            if ev > 0:
                parlays.append({
                    "legs": n_legs,
                    "bets": [b["bet"] for b in combo],
                    "games": [b["game"] for b in combo],
                    "combined_ml": combined_ml,
                    "combined_prob": round(combined_prob, 4),
                    "ev": round(ev, 4),
                    "confidence": confidence_stars(ev),
                    "leg_evs": [b["ev"] for b in combo],
                })

    return sorted(parlays, key=lambda x: x["ev"], reverse=True)[:20]


# ── Page render ───────────────────────────────────────────────────────────────

def render(season: str):
    st.title("💰 Value Bets & Parlays")
    st.caption("Ranked by Expected Value — edges between our model and sportsbook lines")

    api_key = odds_api_key_widget()

    if not api_key:
        st.warning(
            "Add your free Odds API key in the sidebar to unlock this page.\n\n"
            "Get one free at [the-odds-api.com](https://the-odds-api.com) — 500 requests/month."
        )
        st.info(
            "Without a key, this page can't compare our lines to sportsbook lines to find edges. "
            "All other pages still work and show our model lines."
        )
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading model data..."):
        try:
            team_stats, team_adv, elo_ratings, all_teams = load_game_model_data(season)
        except Exception as e:
            st.error(f"Could not load model data: {e}")
            return

    with st.spinner("Loading today's games..."):
        try:
            today_games = get_todays_games()
        except Exception:
            today_games = pd.DataFrame()

    with st.spinner("Fetching sportsbook odds..."):
        raw_odds = get_game_odds(api_key)
        if raw_odds and isinstance(raw_odds[0], dict) and "error" in raw_odds[0]:
            st.error(f"Odds API error: {raw_odds[0]['error']}")
            return
        market_lines = parse_game_odds(raw_odds) if raw_odds else {}

    with st.spinner("Fetching player props..."):
        sportsbook_props = get_all_player_props(api_key)

    team_lookup = {row["id"]: row for _, row in all_teams.iterrows()}

    # ── Scan for edges ────────────────────────────────────────────────────────
    with st.spinner("Scanning for value bets..."):
        game_bets = scan_game_bets(today_games, team_stats, team_adv, elo_ratings, team_lookup, market_lines, season)
        prop_bets = scan_prop_bets(sportsbook_props, season)
        all_bets = sorted(game_bets + prop_bets, key=lambda x: x["ev"], reverse=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        show_category = st.selectbox("Filter by type", ["All", "Game (Moneyline)", "Player Props"])
    with col_filter2:
        min_stars = st.selectbox("Min confidence", ["⭐ (all)", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
    with col_filter3:
        show_parlays = st.checkbox("Show parlay builder", value=True)

    # Apply filters
    filtered = all_bets
    if show_category == "Game (Moneyline)":
        filtered = [b for b in filtered if b["category"] == "game"]
    elif show_category == "Player Props":
        filtered = [b for b in filtered if b["category"] == "prop"]
    min_star_count = min_stars.count("⭐")
    filtered = [b for b in filtered if b["confidence"].count("⭐") >= min_star_count]

    # ── Value bets table ──────────────────────────────────────────────────────
    st.markdown("---")
    if not filtered:
        st.info("No value bets found today above the minimum edge threshold. Try lowering the confidence filter or check back after lines move.")
    else:
        st.markdown(f"### 🎯 Top Value Bets — {len(filtered)} found")
        st.caption(f"Min edge: {MIN_EDGE_PCT}% | Min EV: {MIN_EV:.0%} per $1")

        for i, bet in enumerate(filtered, 1):
            ev_color = "#2ecc71" if bet["ev"] >= 0.08 else "#f39c12" if bet["ev"] >= 0.04 else "#e67e22"
            with st.container():
                c_rank, c_conf, c_bet, c_our, c_book, c_edge, c_ev = st.columns([0.5, 1.5, 3, 1.5, 1.5, 1.5, 1.5])
                c_rank.markdown(f"**#{i}**")
                c_conf.write(bet["confidence"])
                c_bet.markdown(f"**{bet['bet']}**  \n<small style='color:#aaa'>{bet['game']}</small>", unsafe_allow_html=True)
                c_our.metric("Our ML", fmt_moneyline(bet["our_ml"]))
                c_book.metric(f"{bet.get('book','Book')} ML", fmt_moneyline(bet["book_ml"]))
                c_edge.metric("Edge", f"+{bet['edge_pct']:.1f}%")
                c_ev.markdown(
                    f"<div style='background:{ev_color}22;border:1px solid {ev_color};border-radius:6px;"
                    f"padding:6px 10px;text-align:center'>"
                    f"<div style='color:{ev_color};font-size:1.2rem;font-weight:800'>+{bet['ev']*100:.1f}¢</div>"
                    f"<div style='font-size:0.75rem;color:#aaa'>EV per $1</div></div>",
                    unsafe_allow_html=True,
                )
            if i < len(filtered):
                st.markdown("<hr style='margin:6px 0;opacity:0.2'>", unsafe_allow_html=True)

    # ── Parlay builder ────────────────────────────────────────────────────────
    if show_parlays and filtered:
        st.markdown("---")
        st.markdown("### 🎰 Best Parlay Combinations")
        st.caption("Built from top value bets above — ranked by combined expected value")

        with st.spinner("Building parlays..."):
            parlays = build_parlays(filtered)

        if not parlays:
            st.info("Need at least 2 value bets to build parlays.")
        else:
            tabs = st.tabs([f"{p['legs']}-Leg Parlays" for p in parlays[:6]])
            for tab, parlay in zip(tabs, parlays[:6]):
                with tab:
                    st.markdown(f"**Parlay Payout: {fmt_moneyline(parlay['combined_ml'])}**")
                    st.caption(f"Combined win prob: {parlay['combined_prob']*100:.1f}%  |  EV: +{parlay['ev']*100:.1f}¢ per $1  |  {parlay['confidence']}")
                    for leg_bet, leg_game, leg_ev in zip(parlay["bets"], parlay["games"], parlay["leg_evs"]):
                        st.markdown(f"• **{leg_bet}** — {leg_game}  *(leg EV: +{leg_ev*100:.1f}¢)*")

            # Full parlay table
            with st.expander("All parlays table"):
                rows = []
                for p in parlays:
                    rows.append({
                        "Legs": p["legs"],
                        "Bets": " + ".join(p["bets"]),
                        "Payout ML": fmt_moneyline(p["combined_ml"]),
                        "Win Prob": f"{p['combined_prob']*100:.1f}%",
                        "EV per $1": f"+{p['ev']*100:.1f}¢",
                        "Confidence": p["confidence"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
