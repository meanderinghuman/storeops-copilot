"""Seasonal-naive quantile forecast: the stand-in for ShelfSense until Project 2 exists.

p50 for a future day = mean of the same weekday over the last 4 weeks.
Spread = standard deviation of the same weekday over the last 8 weeks; p10/p90 = p50 -/+ 1.2816 sd
(normal approximation, p10 floored at 0). Horizon totals sum the daily means and combine the
daily spreads assuming independent days.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta

Z90 = 1.2815515655446004  # standard normal 90th percentile
MEAN_WEEKS = 4
SPREAD_WEEKS = 8
MAX_HORIZON_DAYS = 28
MODEL_NAME = "seasonal_naive_4w"


class UnknownSeries(LookupError):
    pass


def _quantiles(mean: float, sd: float) -> dict[str, float]:
    return {
        "p10": round(max(0.0, mean - Z90 * sd), 1),
        "p50": round(mean, 1),
        "p90": round(mean + Z90 * sd, 1),
    }


def seasonal_naive(conn: sqlite3.Connection, item_id: str, store_id: str, days: int) -> dict:
    """Forecast `days` days after the database's as-of date, in the ShelfSense response format."""
    if not 1 <= days <= MAX_HORIZON_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_HORIZON_DAYS}")
    as_of = date.fromisoformat(
        conn.execute("SELECT value FROM meta WHERE key = 'as_of_date'").fetchone()[0]
    )
    first = as_of - timedelta(days=7 * SPREAD_WEEKS - 1)
    rows = conn.execute(
        "SELECT date, units FROM sales WHERE item_id = ? AND store_id = ? AND date >= ?",
        (item_id, store_id, first.isoformat()),
    ).fetchall()
    if not rows:
        raise UnknownSeries(f"no sales history for item {item_id} at store {store_id}")
    history = {date.fromisoformat(d): u for d, u in rows}

    daily, total_mean, total_var = [], 0.0, 0.0
    for k in range(1, days + 1):
        day = as_of + timedelta(days=k)
        # The most recent observed same-weekday days, newest first: step back whole weeks
        # starting from the first week that lands on or before the as-of date.
        w0 = math.ceil(k / 7)
        same_wd = [
            history.get(day - timedelta(days=7 * w), 0) for w in range(w0, w0 + SPREAD_WEEKS)
        ]
        mean = sum(same_wd[:MEAN_WEEKS]) / MEAN_WEEKS
        mu_all = sum(same_wd) / len(same_wd)
        sd = math.sqrt(sum((x - mu_all) ** 2 for x in same_wd) / (len(same_wd) - 1))
        daily.append({"date": day.isoformat(), **_quantiles(mean, sd)})
        total_mean += mean
        total_var += sd * sd

    return {
        "item_id": item_id,
        "store_id": store_id,
        "horizon_days": days,
        "start_date": daily[0]["date"],
        "end_date": daily[-1]["date"],
        "model": MODEL_NAME,
        "total": _quantiles(total_mean, math.sqrt(total_var)),
        "daily": daily,
    }
