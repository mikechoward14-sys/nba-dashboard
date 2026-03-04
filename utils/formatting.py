"""Display helpers for moneylines, spreads, and colors."""

def fmt_moneyline(ml: int) -> str:
    if ml > 0:
        return f"+{ml}"
    return str(ml)

def fmt_spread(spread: float) -> str:
    if spread > 0:
        return f"+{spread:.1f}"
    return f"{spread:.1f}"

def spread_display(home_spread: float) -> tuple[str, str]:
    """Returns (home_spread_str, away_spread_str)."""
    return fmt_spread(-home_spread), fmt_spread(home_spread)

def prob_color(prob: float) -> str:
    """Return a hex color: green for high prob, red for low."""
    if prob >= 0.65:
        return "#2ecc71"
    elif prob >= 0.55:
        return "#f39c12"
    else:
        return "#e74c3c"

def ml_badge(ml: int) -> str:
    sign = "+" if ml > 0 else ""
    return f"{sign}{ml}"

def hit_rate_label(rate: float) -> str:
    pct = rate * 100
    if pct >= 65:
        return f"🔥 {pct:.0f}%"
    elif pct >= 50:
        return f"✅ {pct:.0f}%"
    else:
        return f"❌ {pct:.0f}%"
