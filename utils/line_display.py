"""
Reusable side-by-side line display components.
Shows Our Line vs Sportsbook Line with edge highlighting.
"""
import os
import streamlit as st

def _edge_color(diff: float, good_threshold: float = 3.0) -> str:
    if diff >= good_threshold:
        return "#2ecc71"   # green — we like the over/home side
    elif diff <= -good_threshold:
        return "#e74c3c"   # red — we like the other side
    return "#95a5a6"       # gray — roughly fair

def _ml_diff_label(our_ml: int, book_ml: int) -> tuple[str, str]:
    """Returns (label, css_color) for moneyline edge."""
    if our_ml == 0 or book_ml == 0:
        return "—", "#95a5a6"
    # Convert both to implied prob
    def to_prob(ml):
        if ml > 0:
            return 100 / (ml + 100)
        return abs(ml) / (abs(ml) + 100)
    our_prob = to_prob(our_ml)
    book_prob = to_prob(book_ml)
    diff_pct = (our_prob - book_prob) * 100
    if abs(diff_pct) < 2:
        return "≈ Fair", "#95a5a6"
    if diff_pct > 0:
        return f"↑ +{diff_pct:.1f}%", "#2ecc71"
    return f"↓ {diff_pct:.1f}%", "#e74c3c"

def _fmt_ml(ml: int | None) -> str:
    if ml is None:
        return "—"
    return f"+{ml}" if ml > 0 else str(ml)

def _fmt_spread(s: float | None) -> str:
    if s is None:
        return "—"
    return f"+{s:.1f}" if s > 0 else f"{s:.1f}"


def game_lines_comparison(
    home_name: str,
    away_name: str,
    our_home_ml: int,
    our_away_ml: int,
    our_spread: float,
    our_total: float,
    book_lines: dict | None = None,
    compact: bool = False,
):
    """
    Renders a side-by-side table of our model lines vs sportsbook lines.
    book_lines: dict from odds_fetcher.find_game_odds() or None if no API key.
    compact: use smaller layout (for Today's Games cards).
    """
    has_book = book_lines and book_lines.get("home_ml") is not None
    book_name = book_lines.get("book", "Sportsbook") if has_book else "Sportsbook"

    bk_home_ml  = book_lines.get("home_ml")  if has_book else None
    bk_away_ml  = book_lines.get("away_ml")  if has_book else None
    bk_spread   = book_lines.get("home_spread") if has_book else None
    bk_total    = book_lines.get("total")    if has_book else None

    rows = [
        ("Home ML",    _fmt_ml(our_home_ml),      _fmt_ml(bk_home_ml),   our_home_ml,  bk_home_ml,  "ml"),
        ("Away ML",    _fmt_ml(our_away_ml),      _fmt_ml(bk_away_ml),   our_away_ml,  bk_away_ml,  "ml"),
        ("Spread (H)", _fmt_spread(-our_spread),   _fmt_spread(bk_spread), -our_spread, bk_spread,   "spread"),
        ("Total O/U",  f"{our_total:.1f}",         f"{bk_total:.1f}" if bk_total else "—", our_total, bk_total, "total"),
    ]

    header_cols = st.columns([2, 2, 2, 2])
    header_cols[0].markdown("**Line**")
    header_cols[1].markdown(f"**Our Model**")
    header_cols[2].markdown(f"**{book_name}**" if has_book else "**Sportsbook**")
    header_cols[3].markdown("**Edge**")

    for label, our_val, bk_val, our_raw, bk_raw, kind in rows:
        c0, c1, c2, c3 = st.columns([2, 2, 2, 2])
        c0.write(label)
        c1.markdown(f"**{our_val}**")
        c2.write(bk_val if has_book else "*(no key)*")

        if has_book and bk_raw is not None:
            if kind == "ml":
                edge_label, color = _ml_diff_label(int(our_raw), int(bk_raw))
            elif kind == "spread":
                diff = float(our_raw or 0) - float(bk_raw or 0)
                if abs(diff) < 0.5:
                    edge_label, color = "≈ Fair", "#95a5a6"
                else:
                    edge_label = f"{'↑' if diff > 0 else '↓'} {diff:+.1f}"
                    color = "#2ecc71" if diff > 0 else "#e74c3c"
            elif kind == "total":
                diff = float(our_raw or 0) - float(bk_raw or 0)
                if abs(diff) < 1.0:
                    edge_label, color = "≈ Fair", "#95a5a6"
                else:
                    edge_label = f"{'⬆' if diff > 0 else '⬇'} {diff:+.1f}"
                    color = "#f39c12"
            else:
                edge_label, color = "—", "#95a5a6"
            c3.markdown(f"<span style='color:{color};font-weight:700'>{edge_label}</span>", unsafe_allow_html=True)
        else:
            c3.write("—")


def prop_line_comparison(
    stat_label: str,
    our_line: float,
    our_proj: float,
    our_over_ml: int,
    our_under_ml: int,
    our_hit_rate: float,
    book_data: dict | None = None,
):
    """
    Side-by-side prop display row.
    book_data: {line, over_ml, under_ml, book} from odds_fetcher, or None.
    """
    has_book = book_data and book_data.get("line") is not None
    book_name = book_data.get("book", "Book") if has_book else "Book"

    bk_line     = book_data.get("line")     if has_book else None
    bk_over_ml  = book_data.get("over_ml")  if has_book else None
    bk_under_ml = book_data.get("under_ml") if has_book else None

    # Edge: if our line differs from book line
    line_diff = (our_line - bk_line) if (has_book and bk_line is not None) else None

    c_stat, c_our_line, c_bk_line, c_our_o, c_bk_o, c_our_u, c_bk_u, c_edge = st.columns([2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.5])

    c_stat.write(f"**{stat_label}**")
    c_our_line.markdown(f"**{our_line}**")
    c_bk_line.write(f"{bk_line:.1f}" if bk_line is not None else "—")
    c_our_o.write(_fmt_ml(our_over_ml))
    c_bk_o.write(_fmt_ml(bk_over_ml) if has_book else "—")
    c_our_u.write(_fmt_ml(our_under_ml))
    c_bk_u.write(_fmt_ml(bk_under_ml) if has_book else "—")

    if line_diff is not None:
        if abs(line_diff) < 0.5:
            edge_label, color = "≈ Same", "#95a5a6"
        elif line_diff > 0:
            edge_label = f"Our: +{line_diff:.1f} higher"
            color = "#f39c12"
        else:
            edge_label = f"Our: {line_diff:.1f} lower"
            color = "#f39c12"
        # Over/under ML edge
        if bk_over_ml and our_over_ml:
            ml_edge, ml_color = _ml_diff_label(our_over_ml, bk_over_ml)
            c_edge.markdown(
                f"<span style='color:{ml_color};font-size:0.85rem'>{ml_edge}</span>",
                unsafe_allow_html=True
            )
        else:
            c_edge.markdown(f"<span style='color:{color};font-size:0.85rem'>{edge_label}</span>", unsafe_allow_html=True)
    else:
        hit_pct = our_hit_rate * 100
        hit_color = "#2ecc71" if hit_pct >= 60 else "#e74c3c" if hit_pct < 45 else "#f39c12"
        c_edge.markdown(f"<span style='color:{hit_color}'>{hit_pct:.0f}% hist</span>", unsafe_allow_html=True)


def prop_table_header():
    """Render column headers for prop_line_comparison rows."""
    c_stat, c_our_line, c_bk_line, c_our_o, c_bk_o, c_our_u, c_bk_u, c_edge = st.columns([2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.5])
    c_stat.markdown("**Stat**")
    c_our_line.markdown("**Our Line**")
    c_bk_line.markdown("**Book Line**")
    c_our_o.markdown("**Our O ML**")
    c_bk_o.markdown("**Book O ML**")
    c_our_u.markdown("**Our U ML**")
    c_bk_u.markdown("**Book U ML**")
    c_edge.markdown("**Edge**")
    st.markdown("<hr style='margin:2px 0 8px 0'>", unsafe_allow_html=True)


def odds_api_key_widget() -> str:
    """
    Sidebar widget for Odds API key. Returns the key string (may be empty).
    Persists in session state.
    """
    if "odds_api_key" not in st.session_state:
        st.session_state.odds_api_key = os.getenv("ODDS_API_KEY", "")

    with st.sidebar.expander("📡 Sportsbook Lines", expanded=False):
        st.caption("Optional: connect live sportsbook odds for side-by-side comparison.")
        key = st.text_input(
            "Odds API Key",
            value=st.session_state.odds_api_key,
            type="password",
            key="odds_key_input",
            placeholder="Free key: the-odds-api.com",
        )
        if key != st.session_state.odds_api_key:
            st.session_state.odds_api_key = key
        if key:
            st.success("Key set — showing vs sportsbook")
        else:
            st.info("No key — showing model lines only")
    return st.session_state.odds_api_key
