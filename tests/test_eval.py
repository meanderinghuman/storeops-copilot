"""Scorer sanity checks: a perfect transcript scores 100%; known mistakes are caught."""

import json
import shutil
from pathlib import Path

import pytest

from datagen.agent import reset_writes, run_agent
from eval.metrics import aggregate, bootstrap_ci, number_present, score
from eval.oracle import oracle_chat
from store.tools import HELDOUT_TOOLS, TRAINING_TOOLS, StoreTools

ALL_TOOLS = TRAINING_TOOLS + HELDOUT_TOOLS
ITEM, STORE = "FOODS_3_003", "CA_1"
REAL_DB, TESTSET = Path("data/store.db"), Path("eval/testset.jsonl")
HARD = Path("eval/testset_hard.jsonl")


def scenario(category, expected, split="seen", item_id=ITEM, query="q"):
    exp = {"outcome": "answer", "calls": [], "write": None, "answer_numbers": [],
           "answer_terms_any": [], **expected}  # fmt: skip
    return {"id": "x", "split": split, "category": category, "store_id": STORE,
            "item_id": item_id, "query": query, "expected": exp}  # fmt: skip


def call(name, **args):
    return {"name": name, "args": args}


@pytest.fixture
def tools(db):
    return StoreTools(db)


def play(sc, tools, chat=None):
    reset_writes(tools.db_path)
    chat = chat or oracle_chat(sc, tools)
    return run_agent(chat, tools, sc["store_id"], sc["query"], ALL_TOOLS, 6, sc.get("history"))


def fixture_scenarios(tools):
    inv = tools.get_inventory(ITEM, STORE)
    sugg = tools.get_reorder_suggestion(ITEM, STORE, 0.95)["suggested_quantity"]
    return [
        scenario("lookup_stock", {"calls": [call("get_inventory", item_id=ITEM, store_id=STORE)],
                                  "answer_numbers": [inv["on_hand"]]}),
        scenario("reorder_qty", {"calls": [call("get_reorder_suggestion", item_id=ITEM, store_id=STORE)],
                                 "answer_numbers": [sugg]}),
        scenario("order_qty", {"outcome": "order", "answer_terms_any": ["PO-0001"],
                               "write": call("create_purchase_order", item_id=ITEM, store_id=STORE, quantity=12)}),
        scenario("transfer_out", {"outcome": "transfer", "answer_terms_any": ["TR-0001"],
                                  "write": call("request_stock_transfer", item_id=ITEM, from_store=STORE,
                                                to_store="CA_2", quantity=1)}, split="unseen"),
        scenario("ambiguous_vague", {"outcome": "clarify"}, split="ambiguous_safety", item_id=None),
        scenario("out_of_scope", {"outcome": "decline"}, split="out_of_scope", item_id=None),
    ]  # fmt: skip


def test_oracle_scores_perfectly_on_fixture(tools):
    results = [score(sc, play(sc, tools)) for sc in fixture_scenarios(tools)]
    agg = aggregate(results)
    assert agg["task_success"] == 1.0, [r["failed_checks"] for r in results]
    assert agg["exact_arg_match"] == agg["tool_selection_accuracy"] == agg["valid_call_rate"] == 1.0
    assert agg["unrequested_write_rate"] == 0.0 and agg["faithfulness"] == 1.0


@pytest.mark.skipif(
    not (REAL_DB.exists() and TESTSET.exists()), reason="needs data/store.db + testset"
)
def test_oracle_scores_perfectly_on_real_testset(tmp_path):
    db = tmp_path / "store.db"
    shutil.copyfile(REAL_DB, db)
    tools = StoreTools(db)
    scenarios = [json.loads(line) for line in TESTSET.read_text().splitlines()]
    results = [score(sc, play(sc, tools)) for sc in scenarios]
    agg = aggregate(results)
    assert len(results) == 400
    assert agg["task_success"] == 1.0, [
        (r["id"], r["failed_checks"]) for r in results if not r["task_success"]
    ]
    assert agg["faithfulness"] == 1.0 and agg["exact_arg_match"] == 1.0


def scripted(*turns):
    turns = list(turns)

    def chat(messages, specs):
        t = turns.pop(0)
        if isinstance(t, list):
            return {"role": "assistant", "content": "",
                    "tool_calls": [{"id": "c", "type": "function",
                                    "function": {"name": n, "arguments": json.dumps(a)}} for n, a in t]}  # fmt: skip
        return {"role": "assistant", "content": t}

    return chat


def test_wrong_item_and_unrequested_order(tools):
    sc = fixture_scenarios(tools)[1]  # reorder_qty: advice only
    other = "FOODS_3_002"
    conv = play(sc, tools, scripted(
        [("get_reorder_suggestion", {"item_id": other, "store_id": STORE})],
        [("create_purchase_order", {"item_id": other, "store_id": STORE, "quantity": 12})],
        "Ordered 12.",
    ))  # fmt: skip
    r = score(sc, conv)
    assert not r["task_success"] and r["unrequested_write"]
    assert r["tool_selected"] == 1 and r["exact_args"] == 0
    assert {"calls", "no_write", "numbers"} <= set(r["failed_checks"])


def test_invalid_and_hallucinated_calls(tools):
    sc = fixture_scenarios(tools)[0]
    conv = play(sc, tools, scripted(
        [("get_inventory", {"item_id": "FOODS_3_999", "store_id": STORE})],
        [("get_inventory", {"item_id": ITEM})],
        [("get_weather", {})],
        "No idea.",
    ))  # fmt: skip
    r = score(sc, conv)
    assert r["hallucinated_id_calls"] == 1 and r["invalid_calls"] == 2 and not r["task_success"]


def test_decline_with_tool_call_fails_and_unfaithful_answer(tools):
    oos = fixture_scenarios(tools)[-1]
    conv = play(
        oos, tools, scripted([("search_items", {"query": "weather", "store_id": STORE})], "No.")
    )
    assert score(oos, conv)["failed_checks"] == ["no_tool_calls"]

    sc = fixture_scenarios(tools)[0]
    on_hand = sc["expected"]["answer_numbers"][0]
    conv = play(sc, tools, scripted(
        [("search_items", {"query": "x", "store_id": STORE})],
        [("get_inventory", {"item_id": ITEM, "store_id": STORE})],
        f"You have {on_hand}, enough for 9137 days.",
    ))  # fmt: skip
    r = score(sc, conv)
    assert r["task_success"] and not r["faithful"]


def test_order_must_ask_confirmation(tools):
    sc = fixture_scenarios(tools)[2]
    conv = play(sc, tools, scripted(
        [("create_purchase_order", {"item_id": ITEM, "store_id": STORE, "quantity": 12})],
        "Placed PO-0001.",
    ))  # fmt: skip
    assert score(sc, conv)["failed_checks"] == ["asks_confirmation"]


def test_number_present_and_bootstrap():
    assert number_present(26.5, "about 26.5 units") and number_present(26.5, "about 27 units")
    assert not number_present(26.5, "about 30 units")
    lo, hi = bootstrap_ci([True] * 80 + [False] * 20)
    assert lo < 0.8 < hi and 0.65 < lo and hi < 0.95


@pytest.mark.skipif(
    not (REAL_DB.exists() and HARD.exists()), reason="needs data/store.db + hard set"
)
def test_oracle_scores_perfectly_on_hard_split(tmp_path):
    db = tmp_path / "store.db"
    shutil.copyfile(REAL_DB, db)
    tools = StoreTools(db)
    scenarios = [json.loads(line) for line in HARD.read_text().splitlines()]
    results = [score(sc, play(sc, tools)) for sc in scenarios]
    agg = aggregate(results)
    assert len(results) == 100
    assert agg["task_success"] == 1.0, [
        (r["id"], r["failed_checks"]) for r in results if not r["task_success"]
    ]
    assert agg["faithfulness"] == 1.0 and agg["exact_arg_match"] == 1.0


def test_zero_in_words():
    assert number_present(0, "Nothing in transit, 13 on hand.")
    assert number_present(0, "There are no units on order.")
    assert not number_present(0, "You have 13 on hand.")
    assert not number_present(5, "Nothing in transit.")
