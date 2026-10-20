"""Automatic checks that decide whether a generated conversation is kept for training.

A conversation is kept only if `check()` returns no rejection reasons. The reasons are counted
to report the retention rate and the most common teacher mistakes.
"""

from __future__ import annotations

import json
import re

from store.tools import WRITE_TOOLS

KINDS = ["lookup", "planning", "action", "ambiguous", "out_of_scope", "advice"]
PLANNING_TOOLS = {"get_forecast", "get_reorder_suggestion", "get_sales_history"}
MAX_ANSWER_CHARS = 1200

_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(%?)")
# Unicode dashes / spaces some models emit (e.g. "PO\u20110001", "12\u202funits").
_PLAIN = str.maketrans(
    {c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"}
    | {c: " " for c in "\u00a0\u2009\u202f"}
)


def plain(text: str) -> str:
    return text.translate(_PLAIN)


def numbers_in(text: str) -> list[tuple[float, bool]]:
    """All numbers in text as (value, is_percent). '1,234' -> 1234.0; IDs yield their digits."""
    return [
        (float(m.group(0).rstrip("%").replace(",", "")), bool(m.group(1)))
        for m in _NUMBER.finditer(text)
    ]


def ungrounded_numbers(answer: str, messages: list[dict]) -> list[float]:
    """Numbers in the answer that appear in no tool result, user message or system prompt.

    A number counts as grounded if it equals a source number, is a source number rounded to an
    integer ("about 13" for 12.8), or is a percentage of a source fraction ("95%" for 0.95).
    """
    sources = {
        v
        for m in messages
        if m["role"] in ("system", "user", "tool")
        for v, _ in numbers_in(m.get("content") or "")
    }
    bad = []
    for v, pct in numbers_in(answer):
        ok = (
            v in sources
            or (v.is_integer() and any(abs(s - v) <= 0.5 for s in sources))
            or (pct and round(v / 100, 4) in sources)
        )
        if not ok:
            bad.append(v)
    return bad


def _args(call: dict) -> dict:
    raw = call["arguments"]
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def check(conv: dict, kind: str) -> list[str]:
    """Return rejection reasons for a conversation of the given request kind (empty = keep)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    reasons = []
    messages, calls = conv["messages"], conv["calls"]
    answer = plain(messages[-1].get("content") or "") if conv["finish"] == "answer" else ""

    if conv["finish"] != "answer":
        reasons.append("max_rounds")
    elif not answer:
        reasons.append("empty_answer")
    if len(answer) > MAX_ANSWER_CHARS:
        reasons.append("too_long")
    if any(not c["ok"] for c in calls):
        reasons.append("tool_error")
    if answer and ungrounded_numbers(answer, messages):
        reasons.append("ungrounded_number")

    # Item IDs must come from an earlier search result, never from the model's imagination.
    found: set[str] = set()
    seen_calls: set[str] = set()
    for c in calls:
        args = _args(c)
        item = args.get("item_id")
        if item and item not in found:
            reasons.append("unsearched_item_id")
        key = c["name"] + json.dumps(args, sort_keys=True)
        if key in seen_calls:
            reasons.append("repeated_call")
        seen_calls.add(key)
        if c["name"] == "search_items" and c["ok"]:
            found.update(m["item_id"] for m in c["result"]["matches"])

    names = [c["name"] for c in calls]
    writes = [c for c in calls if c["name"] in WRITE_TOOLS and c["ok"]]
    if kind == "out_of_scope":
        if calls:
            reasons.append("tool_call_out_of_scope")
    elif kind == "ambiguous":
        if writes:
            reasons.append("order_on_ambiguous")
        if any(n != "search_items" for n in names):
            reasons.append("guessed_tool_call")
        if answer and "?" not in answer:
            reasons.append("no_clarifying_question")
    elif kind == "action":
        nothing_needed = any(
            c["name"] == "get_reorder_suggestion"
            and c["ok"]
            and c["result"]["suggested_quantity"] == 0
            for c in calls
        )
        if not writes:
            if not nothing_needed:
                reasons.append("no_order_staged")
        elif len(writes) > 1:
            reasons.append("multiple_orders")
        elif answer:
            po_id = writes[0]["result"]["po_id"]
            if po_id not in answer:
                reasons.append("po_id_not_reported")
            if not re.search(r"confirm", answer, re.IGNORECASE):
                reasons.append("no_confirmation_request")
    else:  # lookup, planning, advice (advice: mentions ordering but must not order)
        if writes:
            reasons.append("unrequested_order")
        if not calls:
            reasons.append("no_tool_call")
        elif kind in ("planning", "advice") and not PLANNING_TOOLS & set(names):
            reasons.append("planning_without_forecast")
    return sorted(set(reasons))
