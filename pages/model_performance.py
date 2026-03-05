"""
Model Performance page — tracks hit/miss rate on player prop predictions
and shows calibration to continuously improve the model.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from data.tracker import (
    settle_predictions,
    get_performance_summary,
    get_recent_results,
    compute_calibration,
    load_predictions,
    CALIB_FILE,
)
from models.player_props import PROP_CATEGORIES
import json


def render(season: str):
    st.title("📉 Model Performance")
    st.caption("Tracks every player prop prediction, settles results, and recalibrates the model.")

    # ── Auto-settle on load ───────────────────────────────────────────────────
    with st.spinner("Checking for unsettled predictions..."):
        try:
            n_settled = settle_predictions(season)
            if n_settled > 0:
                st.success(f"Settled {n_settled} new predictions and updated calibration.")
        except Exception as e:
            st.warning(f"Could not auto-settle: {e}")

    summary = get_performance_summary()
    recent  = get_recent_results(100)

    # ── Top-level metrics ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", summary["total"])
    c2.metric("Settled", summary["settled"])
    c3.metric("Pending", summary["pending"])

    # Overall hit rate on book lines
    if not recent.empty and "book_line_hit" in recent.columns:
        overall_hit = recent["book_line_hit"].dropna().mean()
        c4.metric("Overall Hit Rate (vs book)", f"{overall_hit*100:.1f}%",
                  delta=f"{(overall_hit - 0.5)*100:+.1f}% vs 50%")
    else:
        c4.metric("Overall Hit Rate", "—")

    st.markdown("---")

    if summary["settled"] == 0:
        st.info(
            "No settled predictions yet. Prop predictions are automatically recorded when you "
            "view the Player Props page. Results settle the next day after games complete."
        )
        st.markdown("### How it works")
        st.markdown("""
        1. **View Player Props** — every prop line you generate is saved with your prediction
        2. **Next day** — this page fetches actual game results from NBA.com and marks predictions won/lost
        3. **Calibration** — if we're systematically over/underconfident on certain stats, the model auto-adjusts
        4. **Improvement** — the std_dev scale factor for each stat gets tuned so probabilities better match reality
        """)
        return

    # ── Per-stat hit rate table ───────────────────────────────────────────────
    st.markdown("### Hit Rate by Stat")

    stat_rows = []
    for stat, data in summary["by_stat"].items():
        stat_rows.append({
            "Stat": PROP_CATEGORIES.get(stat, stat),
            "Predictions": data["count"],
            "Hit Rate (vs book)": f"{data['book_line_hit_rate']*100:.1f}%" if data["book_line_hit_rate"] else "—",
            "Hit Rate (vs our line)": f"{data['our_line_hit_rate']*100:.1f}%" if data["our_line_hit_rate"] else "—",
            "Avg Projected": data["avg_projection"],
            "Avg Actual": data["avg_actual"],
            "Bias": f"{data['avg_actual'] - data['avg_projection']:+.2f}",
        })

    if stat_rows:
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    # ── Hit rate bar chart ────────────────────────────────────────────────────
    if stat_rows:
        fig = go.Figure()
        stats  = [r["Stat"] for r in stat_rows]
        rates  = [float(r["Hit Rate (vs book)"].replace("%",""))/100
                  if r["Hit Rate (vs book)"] != "—" else 0 for r in stat_rows]
        colors = ["#2ecc71" if r >= 0.55 else "#e74c3c" if r < 0.45 else "#f39c12" for r in rates]

        fig.add_trace(go.Bar(x=stats, y=rates, marker_color=colors,
                             text=[f"{r*100:.1f}%" for r in rates], textposition="outside"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="50% baseline")
        fig.update_layout(
            title="Hit Rate vs Book Line by Stat",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Calibration chart ─────────────────────────────────────────────────────
    st.markdown("### Probability Calibration")
    st.caption("A perfectly calibrated model sits on the diagonal. Above = underconfident, below = overconfident.")

    if not recent.empty and "over_prob" in recent.columns and "book_line_hit" in recent.columns:
        cal_df = recent[["over_prob", "book_line_hit"]].dropna().copy()
        cal_df["over_prob"] = pd.to_numeric(cal_df["over_prob"], errors="coerce")
        cal_df["bucket"] = pd.cut(cal_df["over_prob"],
                                   bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                                   labels=["<30%","30-40%","40-50%","50-60%","60-70%","70-80%",">80%"])
        grp = cal_df.groupby("bucket", observed=True).agg(
            predicted=("over_prob", "mean"),
            actual=("book_line_hit", "mean"),
            count=("book_line_hit", "count"),
        ).dropna()

        if not grp.empty:
            fig2 = go.Figure()
            # Perfect calibration line
            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(dash="dash", color="gray"), name="Perfect"))
            # Our calibration
            fig2.add_trace(go.Scatter(
                x=grp["predicted"], y=grp["actual"], mode="markers+lines",
                marker=dict(size=grp["count"].clip(5, 30), color="#3498db"),
                name="Our Model",
                text=[f"n={c}" for c in grp["count"]],
                hovertemplate="Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}",
            ))
            fig2.update_layout(
                xaxis=dict(title="Predicted Probability", tickformat=".0%"),
                yaxis=dict(title="Actual Hit Rate", tickformat=".0%"),
                paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Calibration scale factors ─────────────────────────────────────────────
    st.markdown("### Current Calibration Adjustments")
    if CALIB_FILE.exists():
        scales = json.loads(CALIB_FILE.read_text())
        if scales:
            scale_rows = []
            for stat, scale in scales.items():
                interp = "overconfident → widening distribution" if scale > 1.05 else \
                         "underconfident → tightening distribution" if scale < 0.95 else "well calibrated"
                scale_rows.append({
                    "Stat": PROP_CATEGORIES.get(stat, stat),
                    "Std Dev Scale": f"{scale:.2f}x",
                    "Status": interp,
                })
            st.dataframe(pd.DataFrame(scale_rows), use_container_width=True, hide_index=True)
            st.caption("Scale > 1.0 means we were overconfident — probabilities are being flattened toward 50%.")
        else:
            st.info("Calibration file is empty — need more settled predictions.")
    else:
        st.info(f"Calibration not yet computed. Need {20} settled predictions per stat.")

    # ── Recent results log ────────────────────────────────────────────────────
    st.markdown("### Recent Results")
    if not recent.empty:
        display_cols = [c for c in ["date","player_name","stat","projection","book_line","actual",
                                     "book_line_hit","over_prob","under_prob"] if c in recent.columns]
        display = recent[display_cols].copy()
        display.columns = [c.replace("_"," ").title() for c in display.columns]
        if "Book Line Hit" in display.columns:
            display["Book Line Hit"] = display["Book Line Hit"].map(
                {True: "✅ Over", False: "❌ Under", None: "—"}
            )
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Manual recalibrate ────────────────────────────────────────────────────
    with st.expander("⚙️ Manual Controls"):
        if st.button("Force re-settle all predictions"):
            with st.spinner("Settling..."):
                n = settle_predictions(season)
            st.success(f"Settled {n} predictions")
        if st.button("Recompute calibration"):
            cal = compute_calibration()
            st.success(f"Recalibrated {len(cal)} stats")
            st.json(cal)
        if st.button("Clear all tracking data", type="secondary"):
            from data.tracker import PREDS_FILE
            if PREDS_FILE.exists():
                PREDS_FILE.unlink()
            if CALIB_FILE.exists():
                CALIB_FILE.unlink()
            st.success("Tracking data cleared.")
            st.rerun()
