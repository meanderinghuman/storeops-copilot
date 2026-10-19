import sqlite3
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from store.forecast import UnknownSeries, seasonal_naive
from store.forecast_stub import create_app

ITEM, STORE = "FOODS_3_003", "CA_1"


@pytest.fixture
def conn(built_db):
    c = sqlite3.connect(built_db)
    yield c
    c.close()


def units(conn, d):
    return conn.execute(
        "SELECT units FROM sales WHERE item_id=? AND store_id=? AND date=?", (ITEM, STORE, d)
    ).fetchone()[0]


def test_p50_is_same_weekday_mean_of_last_4_weeks(conn):
    fc = seasonal_naive(conn, ITEM, STORE, 7)
    as_of = date.fromisoformat(conn.execute("SELECT value FROM meta").fetchone()[0])
    for k, day in enumerate(fc["daily"], start=1):
        target = as_of + timedelta(days=k)
        assert day["date"] == target.isoformat()
        past = [units(conn, (target - timedelta(days=7 * w)).isoformat()) for w in range(1, 5)]
        assert day["p50"] == round(sum(past) / 4, 1)


def test_quantiles_ordered_and_totals(conn):
    fc = seasonal_naive(conn, ITEM, STORE, 14)
    assert fc["horizon_days"] == 14 and len(fc["daily"]) == 14
    for q in fc["daily"] + [fc["total"]]:
        assert 0 <= q["p10"] <= q["p50"] <= q["p90"]
    # Daily values are rounded to 0.1, so their sum can drift from the total by 0.05 per day.
    assert fc["total"]["p50"] == pytest.approx(sum(d["p50"] for d in fc["daily"]), abs=0.05 * 14)
    # Week 2 reuses the same observed weeks as week 1, so the same weekday repeats.
    assert fc["daily"][0]["p50"] == fc["daily"][7]["p50"]


def test_horizon_bounds_and_unknown_series(conn):
    seasonal_naive(conn, ITEM, STORE, 28)
    with pytest.raises(ValueError):
        seasonal_naive(conn, ITEM, STORE, 0)
    with pytest.raises(ValueError):
        seasonal_naive(conn, ITEM, STORE, 29)
    with pytest.raises(UnknownSeries):
        seasonal_naive(conn, "FOODS_3_999", STORE, 7)


def test_stub_contract(built_db, conn):
    client = TestClient(create_app(built_db))
    assert client.get("/health").json()["db_exists"] is True
    resp = client.post("/forecast", json={"item_id": ITEM, "store_id": STORE, "horizon_days": 5})
    assert resp.status_code == 200
    assert resp.json() == seasonal_naive(conn, ITEM, STORE, 5)
    assert set(resp.json()) == {
        "item_id",
        "store_id",
        "horizon_days",
        "start_date",
        "end_date",
        "model",
        "total",
        "daily",
    }
    unknown = {"item_id": "NOPE_1_001", "store_id": STORE, "horizon_days": 5}
    assert client.post("/forecast", json=unknown).status_code == 404
    bad = {"item_id": ITEM, "store_id": STORE, "horizon_days": 0}
    assert client.post("/forecast", json=bad).status_code == 422
