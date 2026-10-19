import sqlite3

from store.build_db import build
from store.names import offline_names


def q(db, sql, *args):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def test_stores_and_split(built_db):
    stores = dict(q(built_db, "SELECT store_id, split FROM stores"))
    assert stores == {"CA_1": "train", "CA_2": "train", "CA_3": "test", "CA_4": "test"}


def test_items_sampled_per_dept_and_skip_low_sellers(built_db):
    rows = q(built_db, "SELECT dept_id, COUNT(*) FROM items GROUP BY dept_id")
    assert dict(rows) == {"FOODS_3": 5, "HOUSEHOLD_1": 5}
    # The fixture's _006 items barely sell, so they are not eligible.
    assert not q(built_db, "SELECT 1 FROM items WHERE item_id LIKE '%_006'")


def test_names_unique_and_pack_sizes(built_db):
    names = [r[0] for r in q(built_db, "SELECT name FROM items")]
    assert len(names) == len(set(names)) and all(names)
    packs = dict(q(built_db, "SELECT item_id, pack_size FROM items"))
    assert all(packs[i] in (6, 12, 24) for i in packs if i.startswith("FOODS"))
    assert all(packs[i] in (4, 6, 12) for i in packs if i.startswith("HOUSEHOLD"))


def test_sales_window_and_as_of(built_db):
    (as_of,) = q(built_db, "SELECT value FROM meta WHERE key = 'as_of_date'")[0]
    _lo, hi, n = q(built_db, "SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM sales")[0]
    assert hi == as_of and n == 100
    assert q(built_db, "SELECT COUNT(*) FROM sales")[0][0] == 4 * 10 * 100


def test_inventory_ranges(built_db):
    rows = q(built_db, "SELECT on_hand, on_order, lead_time_days FROM inventory")
    assert len(rows) == 40
    assert all(oh >= 0 and oo >= 0 and 1 <= lt <= 5 for oh, oo, lt in rows)
    # On hand is 3-14 days of the 28-day average demand.
    rows = q(
        built_db,
        "SELECT v.on_hand, AVG(s.units) FROM inventory v JOIN sales s USING (store_id, item_id) "
        "WHERE s.date > date((SELECT value FROM meta WHERE key='as_of_date'), '-28 day') "
        "GROUP BY v.store_id, v.item_id",
    )
    assert all(3 * avg - 1 <= oh <= 14 * avg + 1 for oh, avg in rows)


def test_promotions_in_future_window(built_db):
    as_of = q(built_db, "SELECT value FROM meta WHERE key = 'as_of_date'")[0][0]
    for start, end, pct, promo, regular in q(
        built_db,
        "SELECT p.start_date, p.end_date, p.discount_pct, p.promo_price, r.price "
        "FROM promotions p JOIN prices r USING (store_id, item_id)",
    ):
        assert as_of < start < end
        assert promo < regular and 10 <= pct <= 30


def test_rebuild_is_identical(built_db, tmp_path):
    raw = built_db.parent / "raw"
    again = build(
        raw,
        tmp_path / "again.db",
        depts=("FOODS_3", "HOUSEHOLD_1"),
        items_per_dept=5,
        history_days=100,
        names_mode="offline",
    )
    tables = ["items", "inventory", "promotions", "prices"]
    for t in tables:
        assert q(built_db, f"SELECT * FROM {t} ORDER BY 1, 2") == q(
            again, f"SELECT * FROM {t} ORDER BY 1, 2"
        )


def test_offline_names_deterministic_and_unique():
    ids = [f"HOUSEHOLD_1_{i:03d}" for i in range(1, 120)]
    a, b = offline_names("HOUSEHOLD_1", ids), offline_names("HOUSEHOLD_1", list(reversed(ids)))
    assert a == b
    assert len(set(a.values())) == len(ids)
