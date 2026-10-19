import math
import sqlite3
from statistics import NormalDist

import jsonschema
import pytest
from fastapi.testclient import TestClient

from store.forecast import Z90
from store.forecast_stub import create_app
from store.tools import (
    HELDOUT_TOOLS,
    TOOL_SCHEMAS,
    TRAINING_TOOLS,
    WRITE_TOOLS,
    StoreTools,
    ToolError,
    apply_defaults,
    tool_specs,
)

ITEM, STORE = "FOODS_3_003", "CA_1"


@pytest.fixture
def tools(db):
    return StoreTools(db)


def row(db, sql, *args):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchone()
    finally:
        conn.close()


# -- schemas ---------------------------------------------------------------------------------


def test_schemas_are_valid_and_split():
    assert len(TRAINING_TOOLS) == 6 and len(HELDOUT_TOOLS) == 2
    assert set(TRAINING_TOOLS) | set(HELDOUT_TOOLS) == set(TOOL_SCHEMAS)
    assert WRITE_TOOLS <= set(TOOL_SCHEMAS)
    for spec in TOOL_SCHEMAS.values():
        params = spec["function"]["parameters"]
        jsonschema.Draft202012Validator.check_schema(params)
        assert set(params["required"]) <= set(params["properties"])
        assert spec["function"]["description"]
    assert [s["function"]["name"] for s in tool_specs()] == TRAINING_TOOLS


def test_apply_defaults():
    assert apply_defaults("get_forecast", {"item_id": ITEM, "store_id": STORE}) == {
        "item_id": ITEM,
        "store_id": STORE,
        "days": 7,
    }
    assert (
        apply_defaults("get_forecast", {"item_id": ITEM, "store_id": STORE, "days": 3})["days"] == 3
    )


@pytest.mark.parametrize(
    "name, args, msg",
    [
        ("get_weather", {}, "Unknown tool"),
        ("get_inventory", "{not json", "not valid JSON"),
        ("get_inventory", "[1, 2]", "JSON object"),
        ("get_inventory", {"item_id": ITEM}, "store_id"),
        ("get_inventory", {"item_id": ITEM, "store_id": STORE, "extra": 1}, "extra"),
        ("get_forecast", {"item_id": ITEM, "store_id": STORE, "days": 90}, "days"),
        ("get_forecast", {"item_id": ITEM, "store_id": STORE, "days": "7"}, "days"),
        ("create_purchase_order", {"item_id": ITEM, "store_id": STORE, "quantity": 0}, "quantity"),
        (
            "create_purchase_order",
            {"item_id": ITEM, "store_id": STORE, "quantity": 2.5},
            "quantity",
        ),
        ("get_inventory", {"item_id": "paper towels", "store_id": STORE}, "item_id"),
    ],
)
def test_invalid_calls_raise_tool_error(tools, name, args, msg):
    with pytest.raises(ToolError, match=msg):
        tools.execute(name, args)


@pytest.mark.parametrize(
    "args",
    [
        {"item_id": "FOODS_3_999", "store_id": STORE},  # item doesn't exist
        {"item_id": ITEM, "store_id": "CA_9"},  # store doesn't exist
        {"item_id": ITEM, "store_id": "TX_1"},  # other state, not in the DB
    ],
)
def test_hallucinated_ids_rejected(tools, args):
    for name in ["get_inventory", "get_sales_history", "get_forecast", "get_reorder_suggestion"]:
        with pytest.raises(ToolError, match="Unknown"):
            tools.execute(name, args)


def test_json_string_arguments_accepted(tools):
    out = tools.execute("get_inventory", f'{{"item_id": "{ITEM}", "store_id": "{STORE}"}}')
    assert out["item_id"] == ITEM


# -- read tools ------------------------------------------------------------------------------


def test_search_items(tools, db):
    (name,) = row(db, "SELECT name FROM items WHERE item_id = ?", ITEM)
    first_word = name.split()[0].lower()
    hits = tools.execute("search_items", {"query": first_word, "store_id": STORE})["matches"]
    assert ITEM in [h["item_id"] for h in hits] and len(hits) <= 5
    assert {"item_id", "name", "department", "price"} == set(hits[0])
    # Typo and exact-ID lookups.
    typo = first_word[:-2] + first_word[-1] + first_word[-2]
    assert ITEM in [h["item_id"] for h in tools.search_items(typo, STORE)["matches"]]
    assert tools.search_items(ITEM.lower(), STORE)["matches"][0]["item_id"] == ITEM
    assert tools.search_items("unicorn saddle", STORE)["matches"] == []


def test_get_inventory_matches_db(tools, db):
    out = tools.execute("get_inventory", {"item_id": ITEM, "store_id": STORE})
    on_hand, on_order, lead = row(
        db,
        "SELECT on_hand, on_order, lead_time_days FROM inventory WHERE item_id=? AND store_id=?",
        ITEM,
        STORE,
    )
    assert (out["on_hand"], out["on_order"], out["lead_time_days"]) == (on_hand, on_order, lead)
    assert out["pending_po_quantity"] == 0 and out["pack_size"] > 0


def test_get_sales_history(tools, db):
    out = tools.execute("get_sales_history", {"item_id": ITEM, "store_id": STORE})
    assert len(out["weeks"]) == 4  # default
    assert out["weeks"][-1]["week_end"] == tools.as_of.isoformat()
    (expected,) = row(
        db,
        "SELECT SUM(units) FROM sales WHERE item_id=? AND store_id=? AND date >= ?",
        ITEM,
        STORE,
        out["weeks"][0]["week_start"],
    )
    assert out["total_units"] == expected == sum(w["units"] for w in out["weeks"])
    assert out["avg_units_per_week"] == round(expected / 4, 1)


def test_get_forecast_local_and_http_agree(tools, db, monkeypatch):
    local = tools.execute("get_forecast", {"item_id": ITEM, "store_id": STORE, "days": 5})
    assert local["total"]["p10"] <= local["total"]["p50"] <= local["total"]["p90"]

    client = TestClient(create_app(db))
    monkeypatch.setattr(
        "store.tools.httpx.post", lambda url, json, timeout: client.post("/forecast", json=json)
    )
    remote = StoreTools(db, forecast_url="http://forecast:8001")
    assert remote.execute("get_forecast", {"item_id": ITEM, "store_id": STORE, "days": 5}) == local


def test_forecast_service_down(db):
    down = StoreTools(db, forecast_url="http://127.0.0.1:9")
    with pytest.raises(ToolError, match="unavailable"):
        down.execute("get_forecast", {"item_id": ITEM, "store_id": STORE})


def test_reorder_math(tools):
    out = tools.execute("get_reorder_suggestion", {"item_id": ITEM, "store_id": STORE})
    assert out["service_level"] == 0.95
    assert out["coverage_days"] == out["lead_time_days"] + 7
    fc = tools.get_forecast(ITEM, STORE, out["coverage_days"])["total"]
    sd = (fc["p90"] - fc["p50"]) / Z90
    demand = round(fc["p50"] + NormalDist().inv_cdf(0.95) * sd, 1)
    assert out["demand_at_service_level"] == demand
    available = out["on_hand"] + out["on_order"] + out["pending_po_quantity"]
    assert out["net_need"] == round(demand - available, 1)
    expected = math.ceil(out["net_need"] / out["pack_size"]) * out["pack_size"]
    assert out["suggested_quantity"] == max(0, expected)
    assert out["suggested_quantity"] % out["pack_size"] == 0


def test_reorder_monotonic_in_service_level(tools):
    items = [f"{d}_{n:03d}" for d in ("FOODS_3", "HOUSEHOLD_1") for n in range(1, 6)]
    for item in items:
        qs = [
            tools.get_reorder_suggestion(item, STORE, sl)["suggested_quantity"]
            for sl in (0.5, 0.8, 0.95, 0.99)
        ]
        assert qs == sorted(qs)


def test_reorder_zero_when_overstocked(tools, db):
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE inventory SET on_hand = 10000 WHERE item_id=? AND store_id=?", (ITEM, STORE)
    )
    conn.commit()
    conn.close()
    out = tools.get_reorder_suggestion(ITEM, STORE, 0.99)
    assert out["suggested_quantity"] == 0 and out["packs"] == 0 and out["net_need"] < 0


def test_get_promotions(tools, db):
    out = tools.execute("get_promotions", {"store_id": STORE, "days": 28})
    for p in out["promotions"]:
        assert p["start_date"] <= out["window_end"] and p["end_date"] >= out["window_start"]
        assert p["promo_price"] < p["regular_price"]
    (total,) = row(db, "SELECT COUNT(*) FROM promotions WHERE store_id = ?", STORE)
    assert len(out["promotions"]) == total  # all generated promos start within 21 days


# -- write tools and confirmation ------------------------------------------------------------


def test_purchase_order_is_pending_until_confirmed(tools, db):
    before = tools.get_inventory(ITEM, STORE)
    po = tools.execute(
        "create_purchase_order", {"item_id": ITEM, "store_id": STORE, "quantity": 24}
    )
    assert po["po_id"] == "PO-0001" and po["status"] == "pending"
    assert row(db, "SELECT status FROM purchase_orders WHERE po_id='PO-0001'") == ("pending",)

    mid = tools.get_inventory(ITEM, STORE)
    assert mid["on_order"] == before["on_order"] and mid["pending_po_quantity"] == 24

    assert tools.confirm_purchase_order("PO-0001")["status"] == "confirmed"
    after = tools.get_inventory(ITEM, STORE)
    assert after["on_order"] == before["on_order"] + 24 and after["pending_po_quantity"] == 0
    with pytest.raises(ToolError, match="already confirmed"):
        tools.confirm_purchase_order("PO-0001")


def test_cancel_purchase_order(tools):
    tools.create_purchase_order(ITEM, STORE, 12)
    po = tools.create_purchase_order(ITEM, STORE, 6)
    assert po["po_id"] == "PO-0002"
    assert tools.cancel_purchase_order("PO-0002")["status"] == "cancelled"
    assert tools.get_inventory(ITEM, STORE)["pending_po_quantity"] == 12
    with pytest.raises(ToolError, match="No such"):
        tools.cancel_purchase_order("PO-9999")


def test_model_cannot_confirm(tools):
    for name in ["confirm_purchase_order", "cancel_purchase_order", "confirm_stock_transfer"]:
        with pytest.raises(ToolError, match="Unknown tool"):
            tools.execute(name, {"po_id": "PO-0001"})


def test_purchase_order_rejects_unknown_ids(tools, db):
    with pytest.raises(ToolError):
        tools.create_purchase_order("FOODS_3_999", STORE, 5)
    assert row(db, "SELECT COUNT(*) FROM purchase_orders") == (0,)


def test_reads_are_read_only(tools):
    with tools._read() as conn, pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM inventory")


def test_stock_transfer(tools):
    src = tools.get_inventory(ITEM, STORE)["on_hand"]
    dst = tools.get_inventory(ITEM, "CA_2")
    args = {"item_id": ITEM, "from_store": STORE, "to_store": "CA_2", "quantity": 1}
    tr = tools.execute("request_stock_transfer", args)
    assert tr["transfer_id"] == "TR-0001" and tr["status"] == "pending"
    assert tools.get_inventory(ITEM, STORE)["on_hand"] == src  # nothing moves yet

    tools.confirm_stock_transfer("TR-0001")
    assert tools.get_inventory(ITEM, STORE)["on_hand"] == src - 1
    assert tools.get_inventory(ITEM, "CA_2")["on_order"] == dst["on_order"] + 1

    with pytest.raises(ToolError, match="different"):
        tools.execute("request_stock_transfer", {**args, "to_store": STORE})
    with pytest.raises(ToolError, match="only"):
        tools.execute("request_stock_transfer", {**args, "quantity": src + 100})
