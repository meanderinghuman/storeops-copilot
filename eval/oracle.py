"""A perfect agent for a test scenario: makes exactly the expected calls and writes an ideal answer.

Used to prove the scorer gives 100% on a perfect transcript (the plan's sanity check).
"""

from __future__ import annotations

from store.tools import StoreTools


def oracle_chat(scenario: dict, tools: StoreTools):
    """A chat function for `run_agent` that plays the ideal transcript for this scenario."""
    exp = scenario["expected"]
    store = scenario["store_id"]
    planned: list[list[dict]] = []

    item = scenario.get("item_id")
    if item:
        name = tools.get_inventory(item, store)["name"]
        planned.append([_tc("search_items", {"query": name, "store_id": store})])
    for c in exp["calls"]:
        if c["name"] == "search_items":
            continue
        name = c["name"][0] if isinstance(c["name"], list) else c["name"]
        args = dict(c["args"])
        if name == "get_promotions":
            args.setdefault("days", 14)
        planned.append([_tc(name, args)])
    if exp["write"]:
        planned.append([_tc(exp["write"]["name"], exp["write"]["args"])])

    def answer() -> str:
        parts = [str(n) for n in exp["answer_numbers"]] + exp["answer_terms_any"][:1]
        if exp["outcome"] == "decline":
            return (
                "Sorry, I can only help with this store's inventory, sales, forecasts and ordering."
            )
        if exp["outcome"] == "clarify":
            return "Which product do you mean?"
        text = "Here you go: " + ", ".join(parts) + "." if parts else "Done."
        if exp["outcome"] in ("order", "transfer"):
            text += " Please confirm it in the app."
        return text

    def chat(messages, specs):
        if planned:
            return {"role": "assistant", "content": "", "tool_calls": planned.pop(0)}
        return {"role": "assistant", "content": answer()}

    return chat


def _tc(name: str, args: dict) -> dict:
    import json

    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
