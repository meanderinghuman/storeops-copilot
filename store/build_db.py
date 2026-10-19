"""Build the mock store database (SQLite) from the M5 Forecasting (Walmart) dataset.

Inputs in `data/raw/` (download from kaggle.com/competitions/m5-forecasting-accuracy):
    sales_train_evaluation.csv, calendar.csv, sell_prices.csv

Usage:
    uv run python -m store.build_db                    # names from cache, else teacher API
    uv run python -m store.build_db --names offline    # no API: vocabulary-based names

Sales are real (one state, a few departments, a sample of items). Product names, inventory,
pack sizes, lead times and promotions are synthetic. Everything random uses one seed, so a
rebuild produces an identical database.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from store import names as names_mod

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE stores (
    store_id TEXT PRIMARY KEY,
    state_id TEXT NOT NULL,
    split    TEXT NOT NULL CHECK (split IN ('train', 'test'))
);

CREATE TABLE items (
    item_id   TEXT PRIMARY KEY,
    dept_id   TEXT NOT NULL,
    cat_id    TEXT NOT NULL,
    name      TEXT NOT NULL UNIQUE,
    pack_size INTEGER NOT NULL CHECK (pack_size > 0)
);

CREATE TABLE prices (
    store_id TEXT NOT NULL REFERENCES stores,
    item_id  TEXT NOT NULL REFERENCES items,
    price    REAL NOT NULL,
    PRIMARY KEY (store_id, item_id)
);

CREATE TABLE sales (
    store_id TEXT NOT NULL,
    item_id  TEXT NOT NULL,
    date     TEXT NOT NULL,
    units    INTEGER NOT NULL,
    PRIMARY KEY (store_id, item_id, date)
) WITHOUT ROWID;

CREATE TABLE inventory (
    store_id       TEXT NOT NULL REFERENCES stores,
    item_id        TEXT NOT NULL REFERENCES items,
    on_hand        INTEGER NOT NULL CHECK (on_hand >= 0),
    on_order       INTEGER NOT NULL CHECK (on_order >= 0),
    lead_time_days INTEGER NOT NULL CHECK (lead_time_days BETWEEN 1 AND 5),
    PRIMARY KEY (store_id, item_id)
);

CREATE TABLE promotions (
    promo_id     TEXT PRIMARY KEY,
    store_id     TEXT NOT NULL REFERENCES stores,
    item_id      TEXT NOT NULL REFERENCES items,
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,
    discount_pct INTEGER NOT NULL,
    promo_price  REAL NOT NULL
);
CREATE INDEX promotions_store ON promotions (store_id, start_date);

CREATE TABLE purchase_orders (
    po_id      TEXT PRIMARY KEY,
    item_id    TEXT NOT NULL REFERENCES items,
    store_id   TEXT NOT NULL REFERENCES stores,
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    status     TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'cancelled')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE stock_transfers (
    transfer_id TEXT PRIMARY KEY,
    item_id     TEXT NOT NULL REFERENCES items,
    from_store  TEXT NOT NULL REFERENCES stores,
    to_store    TEXT NOT NULL REFERENCES stores,
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    status      TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'cancelled')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

PACK_SIZES = {"FOODS": [6, 12, 24], "HOUSEHOLD": [4, 6, 12], "HOBBIES": [1, 4, 6]}


def load_m5(raw_dir: Path, state: str, depts: list[str], history_days: int):
    """Return (wide sales for the state/depts limited to the last `history_days`, dates, prices)."""
    calendar = pd.read_csv(raw_dir / "calendar.csv", usecols=["date", "wm_yr_wk", "d"])
    header = pd.read_csv(raw_dir / "sales_train_evaluation.csv", nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")][-history_days:]
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales = pd.read_csv(
        raw_dir / "sales_train_evaluation.csv",
        usecols=id_cols + day_cols,
        dtype={c: "int32" for c in day_cols},
    )
    sales = sales[(sales.state_id == state) & (sales.dept_id.isin(depts))].reset_index(drop=True)
    if sales.empty:
        raise SystemExit(f"no rows for state={state} depts={depts}")
    d_to_date = dict(zip(calendar.d, calendar.date))
    dates = [d_to_date[d] for d in day_cols]
    sales = sales.rename(columns=dict(zip(day_cols, dates)))

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices.store_id.str.startswith(f"{state}_")]
    return sales, dates, prices


def choose_items(sales: pd.DataFrame, dates: list[str], per_dept: int, rng) -> list[str]:
    """Sample items per department, preferring ones that actually sell (avg >= 0.5/day/store)."""
    recent = sales[dates[-90:]].mean(axis=1)
    per_item = recent.groupby([sales.dept_id, sales.item_id]).min()  # weakest store counts
    chosen = []
    for dept in sorted(per_item.index.get_level_values(0).unique()):
        dept_items = per_item.loc[dept].sort_index()
        eligible = sorted(dept_items[dept_items >= 0.5].index)
        if len(eligible) < per_dept:
            eligible = list(dept_items.sort_values(ascending=False).index[:per_dept])
        picked = rng.choice(eligible, size=min(per_dept, len(eligible)), replace=False)
        chosen.extend(sorted(str(i) for i in picked))
    return chosen


def resolve_names(items: pd.DataFrame, mode: str, seed: int) -> tuple[dict[str, str], str]:
    """Item names from cache / teacher / offline vocabulary. Returns (names, source)."""
    if mode == "offline":
        out = {}
        for dept, grp in items.groupby("dept_id"):
            out.update(names_mod.offline_names(dept, list(grp.item_id), seed))
        return out, "offline"

    cache = names_mod.load_cache()
    missing = items[~items.item_id.isin(cache)]
    if not missing.empty:
        try:
            for dept, grp in missing.groupby("dept_id"):
                print(f"naming {len(grp)} {dept} items with the teacher API...")
                cache.update(names_mod.teacher_names(dept, list(grp.item_id), set(cache.values())))
                names_mod.save_cache(cache)
        except Exception as exc:
            if mode == "teacher":
                raise
            print(f"teacher naming unavailable ({exc}); using offline names")
            return resolve_names(items, "offline", seed)
    return {i: cache[i] for i in items.item_id}, "teacher"


def build(
    raw_dir: Path,
    db_path: Path,
    state: str = "CA",
    depts: tuple[str, ...] = ("FOODS_3", "HOUSEHOLD_1", "HOUSEHOLD_2"),
    items_per_dept: int = 100,
    history_days: int = 400,
    test_stores: tuple[str, ...] = ("CA_3", "CA_4"),
    names_mode: str = "auto",
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    sales, dates, prices = load_m5(raw_dir, state, list(depts), history_days)
    as_of = dates[-1]
    store_ids = sorted(sales.store_id.unique())
    unknown_test = set(test_stores) - set(store_ids)
    if unknown_test:
        raise SystemExit(f"test stores not in data: {sorted(unknown_test)}")

    item_ids = choose_items(sales, dates, items_per_dept, rng)
    sales = sales[sales.item_id.isin(item_ids)].sort_values(["store_id", "item_id"])
    items = (
        sales[["item_id", "dept_id", "cat_id"]].drop_duplicates().sort_values("item_id")
    ).reset_index(drop=True)
    names, names_source = resolve_names(items, names_mode, seed)
    items["name"] = items.item_id.map(names)
    items["pack_size"] = [int(rng.choice(PACK_SIZES[c])) for c in items.cat_id]
    pack = dict(zip(items.item_id, items.pack_size))

    # Latest known price per store-item.
    prices = prices[prices.item_id.isin(item_ids)]
    latest = prices.sort_values("wm_yr_wk").groupby(["store_id", "item_id"]).tail(1)

    # Inventory: on hand = 3-14 days of recent (28-day) average demand.
    inv_rows = []
    for row in sales[["store_id", "item_id"] + dates[-28:]].itertuples(index=False):
        store_id, item_id, *recent = row
        avg = float(np.mean(recent))
        on_hand = round(avg * rng.uniform(3, 14))
        on_order = 0
        if rng.random() < 0.15:  # some stock already in transit from earlier orders
            on_order = int(np.ceil(avg * 7 / pack[item_id]) * pack[item_id])
        inv_rows.append((store_id, item_id, on_hand, on_order, int(rng.integers(1, 6))))

    # Promotions over the next 4 weeks for ~8% of store-items.
    price_map = {(r.store_id, r.item_id): r.sell_price for r in latest.itertuples()}
    as_of_ts = pd.Timestamp(as_of)
    promo_rows = []
    for store_id, item_id, *_ in inv_rows:
        if rng.random() >= 0.08 or (store_id, item_id) not in price_map:
            continue
        start = as_of_ts + pd.Timedelta(days=int(rng.integers(1, 22)))
        end = start + pd.Timedelta(days=int(rng.integers(2, 7)))
        pct = int(rng.choice([10, 15, 20, 25, 30]))
        promo_price = round(price_map[(store_id, item_id)] * (100 - pct) / 100, 2)
        promo_rows.append(
            (store_id, item_id, start.date().isoformat(), end.date().isoformat(), pct, promo_price)
        )

    long_sales = sales.melt(
        id_vars=["store_id", "item_id"], value_vars=dates, var_name="date", value_name="units"
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SCHEMA)
        meta = {
            "as_of_date": as_of,
            "seed": seed,
            "state": state,
            "depts": ",".join(depts),
            "history_days": history_days,
            "names_source": names_source,
            "review_period_days": 7,
        }
        conn.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
        conn.executemany(
            "INSERT INTO stores VALUES (?, ?, ?)",
            [(s, state, "test" if s in test_stores else "train") for s in store_ids],
        )
        conn.executemany(
            "INSERT INTO items VALUES (?, ?, ?, ?, ?)",
            items[["item_id", "dept_id", "cat_id", "name", "pack_size"]].itertuples(index=False),
        )
        conn.executemany(
            "INSERT INTO prices VALUES (?, ?, ?)",
            latest[["store_id", "item_id", "sell_price"]].itertuples(index=False),
        )
        conn.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?)",
            ((s, i, d, int(u)) for s, i, d, u in long_sales.itertuples(index=False)),
        )
        conn.executemany("INSERT INTO inventory VALUES (?, ?, ?, ?, ?)", inv_rows)
        conn.executemany(
            "INSERT INTO promotions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(f"PROMO-{n:04d}", *r) for n, r in enumerate(promo_rows, start=1)],
        )
        conn.commit()
    finally:
        conn.close()
    tmp.replace(db_path)

    print(
        f"built {db_path}: {len(store_ids)} stores ({', '.join(test_stores)} held out), "
        f"{len(items)} items, {len(long_sales):,} sales rows through {as_of}, "
        f"{len(promo_rows)} promotions, names={names_source}"
    )
    return db_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--db", type=Path, default=Path("data/store.db"))
    p.add_argument("--state", default="CA")
    p.add_argument("--depts", nargs="+", default=["FOODS_3", "HOUSEHOLD_1", "HOUSEHOLD_2"])
    p.add_argument("--items-per-dept", type=int, default=100)
    p.add_argument("--history-days", type=int, default=400)
    p.add_argument("--test-stores", nargs="+", default=["CA_3", "CA_4"])
    p.add_argument("--names", choices=["auto", "teacher", "offline"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    build(
        a.raw_dir,
        a.db,
        a.state,
        tuple(a.depts),
        a.items_per_dept,
        a.history_days,
        tuple(a.test_stores),
        a.names,
        a.seed,
    )


if __name__ == "__main__":
    main()
