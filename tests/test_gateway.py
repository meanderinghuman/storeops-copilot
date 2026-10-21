"""Gateway safety tests, driven by scripted fake models (no LLM needed)."""

import json

import pytest
from fastapi.testclient import TestClient

from serve.gateway.app import create_app
from serve.gateway.config import Settings
from serve.gateway.copilot import FALLBACK, Copilot
from store.tools import StoreTools

ITEM, STORE = "FOODS_3_003", "CA_1"


def tc(name, **args):
    return {
        "id": f"c_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def script(*turns):
    """A chat factory replaying assistant turns: list = tool calls, str = answer, Exception = raise."""

    def factory(usage):
        queue = list(turns)

        def chat(messages, specs):
            t = queue.pop(0)
            if isinstance(t, Exception):
                raise t
            if isinstance(t, list):
                return {"role": "assistant", "content": "", "tool_calls": t}
            last = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else {}
            try:
                text = t.format(**last) if isinstance(last, dict) else t
            except (KeyError, ValueError, IndexError):  # literal braces, not a template
                text = t
            return {"role": "assistant", "content": text}

        return chat

    return factory


@pytest.fixture
def tools(db):
    return StoreTools(db)


@pytest.fixture
def name(tools):
    return tools.get_inventory(ITEM, STORE)["name"]


def bot(tools, local, escalation=None, **kw):
    s = Settings(escalation_enabled=escalation is not None, **kw)
    return Copilot(s, tools, local=local, escalation=escalation)


def search(name):
    return [tc("search_items", query=name, store_id=STORE)]


def test_happy_path_stays_local(tools, name):
    out = bot(tools, script(search(name), [tc("get_inventory", item_id=ITEM, store_id=STORE)],
                            "You have {on_hand} on hand.")).handle(STORE, "stock?")  # fmt: skip
    assert out["route"] == "local" and not out["escalation_reasons"] and not out["needs_review"]
    assert [t["name"] for t in out["trace"]] == ["search_items", "get_inventory"]


def test_invented_id_is_refused_then_model_recovers(tools, name):
    out = bot(tools, script(
        [tc("get_inventory", item_id=ITEM, store_id=STORE)],  # no search yet: refused
        search(name), [tc("get_inventory", item_id=ITEM, store_id=STORE)], "You have {on_hand}.",
    )).handle(STORE, "stock?")  # fmt: skip
    first = out["trace"][0]
    assert first["ok"] is False and "did not come from search_items" in first["error"]
    assert out["route"] == "local" and out["escalation_reasons"] == []  # recovered


def test_refused_call_escalates_and_cancels_local_writes(tools, name):
    local = script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE, quantity=6)],
                   [tc("get_inventory", item_id="FOODS_3_002", store_id=STORE)],  # unsearched ID
                   "Done.")  # fmt: skip
    big = script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE, quantity=6)],
                 "Staged {po_id}; please confirm it in the app.")  # fmt: skip
    out = bot(tools, local, big).handle(STORE, "order 6")
    assert out["route"] == "escalated" and out["escalation_reasons"] == ["refused_call"]
    assert [p["po_id"] for p in out["pending"]] == ["PO-0002"]
    with tools._read() as conn:
        status = dict(conn.execute("SELECT po_id, status FROM purchase_orders").fetchall())
    assert status == {"PO-0001": "cancelled", "PO-0002": "pending"}  # local attempt rolled back


def test_quantity_cap(tools, name):
    cap = bot(tools, script()).quantity_cap(ITEM, STORE)
    out = bot(tools, script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE,
                                              quantity=cap + 1)], "Too many.")).handle(STORE, "order")  # fmt: skip
    assert out["trace"][1]["ok"] is False and "safety cap" in out["trace"][1]["error"]
    assert out["pending"] == [] and out["needs_review"]  # no escalation configured
    ok = bot(tools, script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE,
                                             quantity=cap)], "Staged {po_id}, confirm it.")).handle(STORE, "order it")  # fmt: skip
    assert ok["pending"][0]["quantity"] == cap


def test_malformed_tool_call_gets_one_retry(tools, name):
    out = bot(tools, script('<tool_call>{"name": broken</tool_call>', search(name),
                            [tc("get_inventory", item_id=ITEM, store_id=STORE)], "On hand: {on_hand}.")
              ).handle(STORE, "stock?")  # fmt: skip
    assert out["route"] == "local" and len(out["trace"]) == 2


def test_round_limit_escalates(tools, name):
    loop = script(*[search(name)] * 6)
    out = bot(tools, loop, script("I can only help with inventory."), max_rounds=2).handle(
        STORE, "?"
    )
    assert out["route"] == "escalated" and out["escalation_reasons"] == ["max_rounds"]


def test_ungrounded_number_escalates(tools, name):
    local = script(
        search(name), [tc("get_inventory", item_id=ITEM, store_id=STORE)], "About 9137 left."
    )
    big = script(
        search(name), [tc("get_inventory", item_id=ITEM, store_id=STORE)], "You have {on_hand}."
    )
    out = bot(tools, local, big).handle(STORE, "stock?")
    assert out["route"] == "escalated" and out["escalation_reasons"] == ["ungrounded_number"]


def test_local_down_escalates_and_both_down_falls_back(tools):
    out = bot(
        tools, script(RuntimeError("llm down")), script("I can only help with inventory.")
    ).handle(STORE, "hi")
    assert out["route"] == "escalated" and out["escalation_reasons"] == ["local_error:RuntimeError"]
    out = bot(tools, script(RuntimeError("down")), script(RuntimeError("down"))).handle(STORE, "hi")
    assert out["route"] == "fallback" and out["answer"] == FALLBACK and out["pending"] == []


def test_model_cannot_confirm_orders(tools):
    out = bot(tools, script([tc("confirm_purchase_order", po_id="PO-0001")], "Confirmed!")).handle(
        STORE, "x"
    )
    assert out["trace"][0]["error"] == "Unknown tool 'confirm_purchase_order'."


# -- HTTP API ---------------------------------------------------------------------------------


def client_for(tools, local, **kw):
    s = Settings(escalation_enabled=False, llm_base_url="http://127.0.0.1:9/v1", **kw)
    return TestClient(create_app(s, Copilot(s, tools, local=local)))


def test_api_chat_confirm_and_metrics(tools, name):
    c = client_for(tools, script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE,
                                                   quantity=6)], "Staged {po_id}. Confirm it in the app."))  # fmt: skip
    r = c.post("/v1/chat", json={"store_id": STORE, "message": "order 6"})
    assert r.status_code == 200 and r.json()["pending"][0]["status"] == "pending"
    assert "conversation" not in r.json()
    turn = r.json()["turn"]  # structured turn for follow-ups: user, tool calls + results, answer
    assert turn[0] == {"role": "user", "content": "order 6"} and turn[-1]["role"] == "assistant"
    assert any(m["role"] == "tool" and m["name"] == "search_items" for m in turn)
    assert [o["po_id"] for o in c.get("/v1/orders").json()] == ["PO-0001"]
    assert c.post("/v1/orders/PO-0001/confirm").json()["status"] == "confirmed"
    assert c.post("/v1/orders/PO-0001/confirm").status_code == 409  # already confirmed
    assert c.get("/v1/orders").json() == []
    m = c.get("/metrics").text
    assert "copilot_requests_total" in m and 'copilot_orders_total{action="confirmed"' in m
    assert c.get("/health").json() == {"status": "ok"}
    ready = c.get("/ready")  # local LLM unreachable and no escalation -> not ready
    assert ready.status_code == 503


def test_api_key_and_rate_limit(tools):
    c = client_for(
        tools,
        script(*["I can only help with inventory."] * 5),
        api_key="secret",
        rate_limit_per_min=2,
    )
    body = {"store_id": STORE, "message": "weather?"}
    assert c.post("/v1/chat", json=body).status_code == 401
    h = {"X-API-Key": "secret"}
    bad = c.post("/v1/chat", json={"store_id": "bad", "message": "x"}, headers=h)
    assert bad.status_code == 422  # validated; counts toward the limit
    assert c.post("/v1/chat", json=body, headers=h).status_code == 200
    assert c.post("/v1/chat", json=body, headers=h).status_code == 429


def test_write_intent_guard(tools, name):
    # Ordering on a stock-check question is refused (the model gets the reason back).
    out = bot(tools, script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE,
                                              quantity=6)], "Done.")).handle(STORE, "can u do a stock check on it?")  # fmt: skip
    assert (
        out["trace"][1]["ok"] is False
        and "did not ask to place an order" in out["trace"][1]["error"]
    )
    # "Send ... to CA_2" must be a transfer, not a purchase order.
    out = bot(tools, script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE,
                                              quantity=2)], "Done.")).handle(STORE, "send 2 to CA_2")  # fmt: skip
    assert "request_stock_transfer" in out["trace"][1]["error"]
    # "yes, go ahead" after the assistant offered to stage an order is a valid order request.
    history = [{"role": "user", "content": "should I order more?"},
               {"role": "assistant", "content": "I'd order 6. Want me to stage it?"}]  # fmt: skip
    out = bot(tools, script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE,
                                              quantity=6)], "Staged {po_id}, confirm it.")
              ).handle(STORE, "yes, go ahead", history)  # fmt: skip
    assert out["trace"][1]["ok"] and out["pending"][0]["quantity"] == 6


def test_intent_rules():
    from serve.gateway.intent import check_write

    assert check_write("create_purchase_order", 24, "X", "Order 24 X.", "") is None
    assert check_write("create_purchase_order", 24, "X", "Need a PO for X", "") is None
    assert check_write("create_purchase_order", 24, "X", "How many X do we have?", "") is not None
    assert (
        check_write("create_purchase_order", 2, "X", "Move 2 X from CA_4 over here.", "")
        is not None
    )
    assert (
        check_write("request_stock_transfer", 2, "X", "Move 2 X from CA_4 over here.", "") is None
    )
    assert check_write("request_stock_transfer", 2, "X", "have 2 X moved from CA_4", "") is None
    assert check_write("request_stock_transfer", 2, "X", "How many X at CA_4?", "") is not None


def test_web_ui_and_listings(tools, name):
    c = client_for(tools, script(search(name), [tc("create_purchase_order", item_id=ITEM, store_id=STORE,
                                                   quantity=6)], "Staged {po_id}. Confirm it in the app."))  # fmt: skip
    page = c.get("/")
    assert page.status_code == 200 and "StoreOps Copilot" in page.text
    for asset in ("/static/app.js", "/static/styles.css"):
        assert c.get(asset).status_code == 200
    c.post("/v1/chat", json={"store_id": STORE, "message": "order 6"})
    orders = c.get("/v1/orders", params={"store_id": STORE}).json()
    assert orders[0]["name"] == name  # listings carry the product name for the UI
    tools.request_stock_transfer(ITEM, STORE, "CA_2", 1)
    assert [
        t["transfer_id"] for t in c.get("/v1/transfers", params={"store_id": "CA_2"}).json()
    ] == ["TR-0001"]
