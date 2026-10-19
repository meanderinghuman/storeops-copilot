"""Test fixtures: a tiny synthetic dataset in the exact M5 file format, built into a store DB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from store.build_db import build

N_DAYS = 120
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1"]
DEPTS = {"FOODS_3": "FOODS", "HOUSEHOLD_1": "HOUSEHOLD"}
ITEMS_PER_DEPT = 6


def write_fake_m5(raw_dir) -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-01-24", periods=N_DAYS, freq="D")
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11600 + i // 7 for i in range(N_DAYS)],
            "d": [f"d_{i + 1}" for i in range(N_DAYS)],
        }
    )
    calendar.to_csv(raw_dir / "calendar.csv", index=False)

    weekday_lift = np.array([1.4, 1.3, 0.9, 0.8, 0.8, 0.9, 1.1])  # Sat..Fri, like M5's wday order
    rows, prices = [], []
    for store in STORES:
        for dept, cat in DEPTS.items():
            for n in range(1, ITEMS_PER_DEPT + 1):
                item = f"{dept}_{n:03d}"
                level = 0.2 if n == ITEMS_PER_DEPT else 2.0 + n  # last item barely sells
                lam = level * weekday_lift[np.arange(N_DAYS) % 7]
                units = rng.poisson(lam)
                rows.append(
                    {
                        "id": f"{item}_{store}_evaluation",
                        "item_id": item,
                        "dept_id": dept,
                        "cat_id": cat,
                        "store_id": store,
                        "state_id": store[:2],
                        **{f"d_{i + 1}": int(u) for i, u in enumerate(units)},
                    }
                )
                for wk in calendar.wm_yr_wk.unique():
                    prices.append((store, item, wk, round(1.5 + n * 0.75, 2)))
    pd.DataFrame(rows).to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        raw_dir / "sell_prices.csv", index=False
    )


@pytest.fixture(scope="session")
def built_db(tmp_path_factory):
    """A store DB built once per session; treat as read-only (use `db` for write tests)."""
    root = tmp_path_factory.mktemp("m5")
    raw = root / "raw"
    raw.mkdir()
    write_fake_m5(raw)
    return build(
        raw,
        root / "store.db",
        depts=tuple(DEPTS),
        items_per_dept=5,
        history_days=100,
        names_mode="offline",
    )


@pytest.fixture
def db(built_db, tmp_path):
    """A fresh writable copy of the store DB for each test."""
    path = tmp_path / "store.db"
    path.write_bytes(built_db.read_bytes())
    return path
