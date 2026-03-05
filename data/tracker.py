"""
Hit/miss tracker for player prop predictions.

Workflow:
  1. record_prediction()   — called when we display a prop line
  2. settle_predictions()  — called on next load; fetches game results from NBA API
                             and marks predictions as won/lost
  3. compute_calibration() — derives per-stat scale factors from settled data
                             and writes .tracking/calibration.json

Storage: .tracking/predictions.jsonl (one JSON object per line)
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TRACKING_DIR  = Path(__file__).parent.parent / ".tracking"
PREDS_FILE    = TRACKING_DIR / "predictions.jsonl"
CALIB_FILE    = TRACKING_DIR / "calibration.json"
TRACKING_DIR.mkdir(exist_ok=True)

MIN_SAMPLES_FOR_CALIBRATION = 20   # need this many settled bets per stat to recalibrate


# ── Record ────────────────────────────────────────────────────────────────────

def record_prediction(
    player_name: str,
    player_id: int,
    stat: str,
    our_line: float,
    book_line: float | None,
    projection: float,
    over_prob: float,
    under_prob: float,
    game_date: str | None = None,
) -> None:
    """Append a prediction to the log file."""
    record = {
        "id": f"{player_id}_{stat}_{game_date or datetime.now().strftime('%Y%m%d')}",
        "player_name": player_name,
        "player_id": player_id,
        "stat": stat,
        "our_line": our_line,
        "book_line": book_line,
        "projection": projection,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "date": game_date or datetime.now().strftime("%Y-%m-%d"),
        "settled": False,
        "actual": None,
        "our_line_hit": None,    # did actual beat our line?
        "book_line_hit": None,   # did actual beat book line?
        "recorded_at": datetime.now().isoformat(),
    }
    # Deduplicate by id
    existing = load_predictions()
    ids = {p["id"] for p in existing}
    if record["id"] not in ids:
        with open(PREDS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")


# ── Load / save ───────────────────────────────────────────────────────────────

def load_predictions() -> list[dict]:
    if not PREDS_FILE.exists():
        return []
    records = []
    with open(PREDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def save_predictions(records: list[dict]) -> None:
    with open(PREDS_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── Settle ────────────────────────────────────────────────────────────────────

def settle_predictions(season: str = "2025-26") -> int:
    """
    For each unsettled prediction from a past date, fetch the player's
    game log and check if the result is available. Returns number settled.
    """
    from data.fetcher import get_player_game_log

    records = load_predictions()
    today = datetime.now().strftime("%Y-%m-%d")
    settled_count = 0

    for rec in records:
        if rec.get("settled"):
            continue
        # Only try to settle predictions from yesterday or earlier
        if rec.get("date", today) >= today:
            continue

        try:
            log = get_player_game_log(int(rec["player_id"]), season=season, last_n=5)
            if log.empty or rec["stat"] not in log.columns:
                continue

            log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
            pred_date = pd.to_datetime(rec["date"])
            # Find game on or after prediction date
            game_rows = log[log["GAME_DATE"] >= pred_date]
            if game_rows.empty:
                continue

            actual = float(pd.to_numeric(game_rows.iloc[0][rec["stat"]], errors="coerce"))
            rec["actual"] = actual
            rec["settled"] = True
            rec["our_line_hit"]  = actual > rec["our_line"]
            rec["book_line_hit"] = actual > rec["book_line"] if rec["book_line"] is not None else None
            rec["settled_at"] = datetime.now().isoformat()
            settled_count += 1
            time.sleep(0.3)
        except Exception:
            continue

    save_predictions(records)
    if settled_count > 0:
        compute_calibration(records)
    return settled_count


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_calibration(records: list[dict] | None = None) -> dict:
    """
    Compute per-stat calibration scale factors.

    For each stat, bucket predictions by predicted probability (e.g. 0.6-0.7).
    If our 65% predictions only hit 50%, the model is overconfident — scale std_dev UP.
    If our 65% predictions hit 75%, the model is underconfident — scale std_dev DOWN.

    Returns {stat: scale_factor} and writes to calibration.json.
    """
    if records is None:
        records = load_predictions()

    settled = [r for r in records if r.get("settled") and r.get("book_line_hit") is not None]
    if not settled:
        return {}

    df = pd.DataFrame(settled)
    calibration = {}

    for stat in df["stat"].unique():
        stat_df = df[df["stat"] == stat].copy()
        if len(stat_df) < MIN_SAMPLES_FOR_CALIBRATION:
            continue

        # For each prediction, we predicted over with over_prob
        # Actual outcome: book_line_hit (True = over won)
        stat_df["predicted_prob"] = stat_df["over_prob"].astype(float)
        stat_df["outcome"] = stat_df["book_line_hit"].astype(float)

        # Brier score: mean((predicted - actual)^2) — lower is better
        brier = ((stat_df["predicted_prob"] - stat_df["outcome"]) ** 2).mean()

        # Calibration: group into probability buckets and compare to actual hit rate
        stat_df["bucket"] = pd.cut(stat_df["predicted_prob"], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
        calib_df = stat_df.groupby("bucket", observed=True).agg(
            predicted=("predicted_prob", "mean"),
            actual=("outcome", "mean"),
            count=("outcome", "count"),
        ).dropna()

        if calib_df.empty:
            continue

        # If predicted avg > actual avg across buckets, we're overconfident
        # → increase std_dev (scale > 1) to flatten probabilities toward 50%
        # If predicted < actual, we're underconfident → decrease std_dev (scale < 1)
        overconfidence = (calib_df["predicted"] - calib_df["actual"]).mean()

        # Scale: each 5% overconfidence adds 10% to std_dev
        scale = 1.0 + (overconfidence / 0.05) * 0.10
        scale = round(max(0.5, min(2.5, scale)), 3)

        calibration[stat] = {
            "scale": scale,
            "brier_score": round(float(brier), 4),
            "sample_size": len(stat_df),
            "overconfidence": round(float(overconfidence), 4),
        }

    # Write flat scale dict for model consumption
    flat = {stat: v["scale"] for stat, v in calibration.items()}
    CALIB_FILE.write_text(json.dumps(flat, indent=2))

    return calibration


# ── Stats summary ─────────────────────────────────────────────────────────────

def get_performance_summary() -> dict:
    """Returns summary stats for the performance dashboard."""
    records = load_predictions()
    if not records:
        return {"total": 0, "settled": 0, "pending": 0}

    df = pd.DataFrame(records)
    settled = df[df["settled"] == True]
    pending = df[df["settled"] == False]

    summary = {
        "total": len(df),
        "settled": len(settled),
        "pending": len(pending),
        "by_stat": {},
    }

    if settled.empty:
        return summary

    for stat in settled["stat"].unique():
        sdf = settled[settled["stat"] == stat]
        book_hits = sdf["book_line_hit"].dropna()
        our_hits  = sdf["our_line_hit"].dropna()

        summary["by_stat"][stat] = {
            "stat_label": stat,
            "count": len(sdf),
            "book_line_hit_rate": round(float(book_hits.mean()), 3) if not book_hits.empty else None,
            "our_line_hit_rate":  round(float(our_hits.mean()),  3) if not our_hits.empty  else None,
            "avg_projection": round(float(sdf["projection"].mean()), 2),
            "avg_actual": round(float(pd.to_numeric(sdf["actual"], errors="coerce").mean()), 2),
        }

    return summary


def get_recent_results(n: int = 50) -> pd.DataFrame:
    """Return the most recent N settled predictions as a DataFrame."""
    records = load_predictions()
    settled = [r for r in records if r.get("settled")]
    settled.sort(key=lambda x: x.get("settled_at", ""), reverse=True)
    return pd.DataFrame(settled[:n]) if settled else pd.DataFrame()
