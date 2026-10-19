"""Agent loop and training-data filters, driven by a scripted fake model (no API calls)."""

import json

import pytest

from datagen.agent import reset_writes, run_agent
from datagen.filters import check, numbers_in, ungrounded_numbers
from store.tools import StoreTools

ITEM, STORE = "FOODS_3_003", "CA_1"


def call(name, **args):
    return {
        "id": f"c_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def scripted(*turns):
    """A fake chat function returning the given assistant turns in order.

    A turn is a list of tool calls, or a string (final answer). Strings may use {key} fields
    filled from the previous tool result (so answers can quote real numbers).
    """
    turns = list(turns)

    def chat(messages, specs):
        turn = turns.pop(0)
        if isinstance(turn, list):
            return {"role": "assistant", "content": "", "tool_calls": turn}
        last = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else {}
        return {"role": "assistant", "content": turn.format(**last)}

    return chat


@pytest.fixture
def tools(db):
    return StoreTools(db)


@pytest.fixture
def name(tools):
    return tools.get_inventory(ITEM, STORE)["name"]


def test_lookup_conversation_kept(tools, name):
    chat = scripted(
        [call("search_items", query=name, store_id=STORE)],
        [call("get_inventory", item_id=ITEM, store_id=STORE)],
        "You have {on_hand} on hand and {on_order} on order.",
    )
    conv = run_agent(chat, tools, STORE, f"How many {name} do we have?")
    assert conv["finish"] == "answer" and conv["rounds"] == 2
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert "Today is Monday, 2016-05-23" in conv["messages"][0]["content"]
    assert check(conv, "lookup") == []


def test_action_conversation_kept(tools, name):
    chat = scripted(
        [call("search_items", query=name, store_id=STORE)],
        [call("create_purchase_order", item_id=ITEM, store_id=STORE, quantity=24)],
        "I staged {po_id} for {quantity} units. Please confirm it in the app.",
    )
    conv = run_agent(chat, tools, STORE, f"Order 24 {name}.")
    assert check(conv, "action") == []
    # The same conversation is an unrequested order if the manager only asked a question.
    assert "unrequested_order" in check(conv, "lookup")


def test_action_rejections(tools, name):
    chat = scripted(
        [call("search_items", query=name, store_id=STORE)],
        [call("create_purchase_order", item_id=ITEM, store_id=STORE, quantity=24)],
        "Done, the order is placed.",
    )
    reasons = check(run_agent(chat, tools, STORE, "Order 24."), "action")
    assert reasons == ["no_confirmation_request", "po_id_not_reported"]


def test_guessed_id_and_tool_error(tools):
    chat = scripted(
        [call("get_inventory", item_id=ITEM, store_id=STORE)],  # no search first
        [call("get_inventory", item_id="FOODS_3_999", store_id=STORE)],  # hallucinated
        "Sorry.",
    )
    conv = run_agent(chat, tools, STORE, "stock?")
    assert conv["calls"][1]["ok"] is False
    assert json.loads(conv["messages"][-2]["content"])["error"].startswith("Unknown item_id")
    assert check(conv, "lookup") == ["tool_error", "unsearched_item_id"]


def test_heldout_tool_not_available_in_training(tools):
    chat = scripted([call("get_promotions", store_id=STORE)], "No promotions.")
    conv = run_agent(chat, tools, STORE, "Any promos?")
    assert conv["calls"][0]["error"] == "Unknown tool 'get_promotions'."


def test_max_rounds(tools, name):
    search = [call("search_items", query=name, store_id=STORE)]
    conv = run_agent(scripted(*[search] * 3), tools, STORE, "?", max_rounds=2)
    assert conv["finish"] == "max_rounds" and conv["rounds"] == 2
    assert {"max_rounds", "repeated_call"} <= set(check(conv, "lookup"))


def test_out_of_scope_and_ambiguous(tools):
    decline = run_agent(scripted("I can only help with store inventory."), tools, STORE, "Weather?")
    assert check(decline, "out_of_scope") == []
    assert check(decline, "lookup") == ["no_tool_call"]

    clarify = run_agent(scripted("Which product do you mean?"), tools, STORE, "Order the usual.")
    assert check(clarify, "ambiguous") == []
    assert check(clarify, "action") == ["no_order_staged"]
    guess = run_agent(scripted("Ordering it now."), tools, STORE, "Order the usual.")
    assert check(guess, "ambiguous") == ["no_clarifying_question"]


def test_ungrounded_numbers():
    msgs = [
        {"role": "system", "content": "Today is 2016-05-23."},
        {"role": "user", "content": "next 5 days"},
        {"role": "tool", "content": '{"p50": 12.8, "service_level": 0.95, "on_hand": 1,234}'},
    ]
    assert ungrounded_numbers("About 13 units (12.8 expected) over 5 days at 95%.", msgs) == []
    assert ungrounded_numbers("You'll be short by 7 units.", msgs) == [7.0]
    assert numbers_in("1,234 units, 95%") == [(1234.0, False), (95.0, True)]


def test_reset_writes(tools, name):
    tools.create_purchase_order(ITEM, STORE, 6)
    reset_writes(tools.db_path)
    assert tools.create_purchase_order(ITEM, STORE, 6)["po_id"] == "PO-0001"


def test_action_with_zero_suggestion_kept(tools, name, db):
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE inventory SET on_hand = 10000 WHERE item_id=? AND store_id=?", (ITEM, STORE)
    )
    conn.commit()
    conn.close()
    chat = scripted(
        [call("search_items", query=name, store_id=STORE)],
        [call("get_reorder_suggestion", item_id=ITEM, store_id=STORE)],
        "No order needed: suggested quantity is {suggested_quantity}.",
    )
    assert check(run_agent(chat, tools, STORE, f"Reorder {name}."), "action") == []


def test_unicode_hyphen_po_id(tools, name):
    chat = scripted(
        [call("search_items", query=name, store_id=STORE)],
        [call("create_purchase_order", item_id=ITEM, store_id=STORE, quantity=24)],
        "Staged PO‑0001 for 24 units. Please confirm it in the app.",
    )
    assert check(run_agent(chat, tools, STORE, f"Order 24 {name}."), "action") == []
