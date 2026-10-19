"""Build the 400-scenario test set (Step 3). Run once, before training, then hand-check.

    uv run python -m eval.build_testset            # writes eval/testset.jsonl + eval/testset_review.csv

Leakage rules: every scenario is at a held-out store (CA_3 / CA_4) that never appears in training
data; the "unseen" split needs the two held-out tools. Expected outcomes are computed from the
store DB (deterministic, seed 42), and GLM-5.3 rewrites each templated request into natural
manager language. A rewrite is only accepted if it keeps the exact product name, numbers and
store IDs; otherwise the template wording is kept.

Scenario: {id, split, category, store_id, query, template, expected, note}
expected = {
  "outcome": "answer" | "order" | "no_order" | "transfer" | "clarify" | "decline",
  "calls":   [{"name": str | [str, ...], "args": {subset that must match}}],  # required reads
  "write":   null | {"name": ..., "args": {exact}},                            # required write
  "answer_numbers": [...],      # every one must appear in the final answer
  "answer_terms_any": [...],    # at least one must appear (case-insensitive)
}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datagen.runner import read_jsonl
from datagen.teacher import Teacher, parse_json_object
from store.tools import StoreTools

STORES = ("CA_3", "CA_4")
OTHER = {"CA_3": "CA_4", "CA_4": "CA_3"}
EXCLUDE = {"FOODS_3_118", "HOUSEHOLD_2_062", "HOUSEHOLD_2_444"}  # near-duplicate names


def _call(name, **args):
    return {"name": name, "args": args}


class Builder:
    def __init__(self, db: Path, seed: int):
        self.t = StoreTools(db)
        self.rng = random.Random(seed)
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT item_id, name FROM items ORDER BY item_id").fetchall()
        conn.close()
        # Only items whose full name finds exactly that item first, so the request is unambiguous.
        self.items = [
            (i, n)
            for i, n in rows
            if i not in EXCLUDE and self.t.search_items(n, "CA_3")["matches"][0]["item_id"] == i
        ]
        self.n = 0
        self.out: list[dict] = []
        self.deck: list[tuple[str, str]] = []

    def pick(self) -> tuple[str, str, str]:
        """Next item from a shuffled deck, so scenarios use different products (refills if empty)."""
        if not self.deck:
            self.deck = self.items[:]
            self.rng.shuffle(self.deck)
        item_id, name = self.deck.pop()
        return item_id, name, STORES[self.n % 2]

    def add(self, split, category, store_id, template, expected, note, item_id=None):
        self.n += 1
        exp = {"outcome": "answer", "calls": [], "write": None, "answer_numbers": [],
               "answer_terms_any": [], **expected}  # fmt: skip
        if item_id is None:  # the product the request is about, if any
            ids = [c["args"]["item_id"] for c in exp["calls"] + [exp["write"] or {"args": {}}]
                   if "item_id" in c["args"]]  # fmt: skip
            item_id = ids[0] if ids else None
        self.out.append({"id": f"t{self.n:03d}", "split": split, "category": category,
                         "store_id": store_id, "item_id": item_id, "query": template,
                         "template": template, "expected": exp, "note": note})  # fmt: skip

    # -- seen tools (150) ---------------------------------------------------------------------

    def seen(self):
        r = self.rng
        for _ in range(25):
            i, n, s = self.pick()
            v = self.t.get_inventory(i, s)
            q = r.choice([f"How many {n} do we have?", f"stock check on {n}",
                          f"What's our on-hand count for {n}?"])  # fmt: skip
            self.add("seen", "lookup_stock", s, q,
                     {"calls": [_call("get_inventory", item_id=i, store_id=s)],
                      "answer_numbers": [v["on_hand"]]},
                     f"Report on hand = {v['on_hand']}.")  # fmt: skip
        for _ in range(8):
            i, n, s = self.pick()
            v = self.t.get_inventory(i, s)
            q = r.choice([f"How many {n} are on order?", f"Do we have any {n} in transit?"])
            self.add("seen", "lookup_on_order", s, q,
                     {"calls": [_call("get_inventory", item_id=i, store_id=s)],
                      "answer_numbers": [v["on_order"]]},
                     f"Report on order = {v['on_order']}.")  # fmt: skip
        for _ in range(7):
            i, n, s = self.pick()
            v = self.t.get_inventory(i, s)
            q = r.choice([f"What's the supplier lead time on {n}?", f"How long does {n} take to arrive if I order today?"])  # fmt: skip
            self.add("seen", "lookup_lead_time", s, q,
                     {"calls": [_call("get_inventory", item_id=i, store_id=s)],
                      "answer_numbers": [v["lead_time_days"]]},
                     f"Report lead time = {v['lead_time_days']} days.")  # fmt: skip
        for _ in range(7):
            i, n, s = self.pick()
            price = next(
                m["price"] for m in self.t.search_items(n, s)["matches"] if m["item_id"] == i
            )
            q = r.choice([f"What's the shelf price of {n}?", f"How much do we charge for {n}?"])
            self.add("seen", "lookup_price", s, q,
                     {"calls": [_call("search_items", store_id=s)], "answer_numbers": [price]},
                     f"Report price = ${price}.", item_id=i)  # fmt: skip
        for _ in range(13):
            i, n, s = self.pick()
            w = r.choice([2, 3, 4, 6, 8])
            h = self.t.get_sales_history(i, s, w)
            q = r.choice([f"How many {n} did we sell over the last {w} weeks?",
                          f"Total sales of {n} for the past {w} weeks?"])  # fmt: skip
            self.add("seen", "lookup_sales", s, q,
                     {"calls": [_call("get_sales_history", item_id=i, store_id=s, weeks=w)],
                      "answer_numbers": [h["total_units"]]},
                     f"Report total units over {w} weeks = {h['total_units']}.")  # fmt: skip
        for _ in range(20):
            i, n, s = self.pick()
            d = r.choice([3, 5, 7, 10, 14])
            f = self.t.get_forecast(i, s, d)["total"]
            q = r.choice([f"How many {n} will we sell in the next {d} days?",
                          f"What's the demand forecast for {n} over the next {d} days?"])  # fmt: skip
            self.add("seen", "forecast_days", s, q,
                     {"calls": [_call("get_forecast", item_id=i, store_id=s, days=d)],
                      "answer_numbers": [f["p50"]]},
                     f"Report expected demand (p50) = {f['p50']} over {d} days (p10 {f['p10']}, p90 {f['p90']}).")  # fmt: skip
        for _ in range(10):
            i, n, s = self.pick()
            v = self.t.get_inventory(i, s)
            q = r.choice([f"Will we run out of {n} before the weekend?",
                          f"Is our stock of {n} enough to get through Sunday?"])  # fmt: skip
            self.add("seen", "runout_weekend", s, q,
                     {"calls": [_call(["get_inventory", "get_reorder_suggestion"], item_id=i, store_id=s),
                                _call(["get_forecast", "get_reorder_suggestion"], item_id=i, store_id=s)],
                      "answer_numbers": [v["on_hand"]]},
                     f"Check stock (on hand {v['on_hand']}) against the forecast; no order.")  # fmt: skip
        for _ in range(15):
            i, n, s = self.pick()
            g = self.t.get_reorder_suggestion(i, s, 0.95)
            q = r.choice([f"How many {n} should I order?", f"What order quantity do you suggest for {n}?"])  # fmt: skip
            self.add("seen", "reorder_qty", s, q,
                     {"calls": [_call("get_reorder_suggestion", item_id=i, store_id=s)],
                      "answer_numbers": [g["suggested_quantity"]]},
                     f"Recommend {g['suggested_quantity']} units; do NOT stage an order.")  # fmt: skip
        for _ in range(25):
            i, n, s = self.pick()
            qty = r.choice([6, 10, 12, 20, 24, 30, 36, 48])
            q = r.choice([f"Order {qty} {n}.", f"Please place an order for {qty} units of {n}.",
                          f"put in a PO for {qty} {n}"])  # fmt: skip
            self.add("seen", "order_qty", s, q,
                     {"outcome": "order",
                      "write": _call("create_purchase_order", item_id=i, store_id=s, quantity=qty),
                      "answer_terms_any": ["PO-0001"]},
                     f"Stage PO for {qty} units, report PO-0001, ask to confirm.")  # fmt: skip
        zero, pos = 0, 0
        while zero + pos < 20:
            i, n, s = self.pick()
            g = self.t.get_reorder_suggestion(i, s, 0.95)["suggested_quantity"]
            if (g == 0 and zero >= 5) or (g > 0 and pos >= 15):
                continue
            q = r.choice([f"Order whatever we need of {n}.", f"Reorder {n}, you work out how many.",
                          f"Place an order for enough {n} to cover us."])  # fmt: skip
            if g:
                pos += 1
                self.add("seen", "order_suggested", s, q,
                         {"outcome": "order",
                          "calls": [_call("get_reorder_suggestion", item_id=i, store_id=s)],
                          "write": _call("create_purchase_order", item_id=i, store_id=s, quantity=g),
                          "answer_terms_any": ["PO-0001"]},
                         f"Suggestion is {g}: stage PO for {g}, ask to confirm.")  # fmt: skip
            else:
                zero += 1
                self.add("seen", "order_suggested_zero", s, q,
                         {"outcome": "no_order",
                          "calls": [_call("get_reorder_suggestion", item_id=i, store_id=s)]},
                         "Suggestion is 0: do NOT order; explain that stock covers demand.")  # fmt: skip

    # -- held-out tools (100) -----------------------------------------------------------------

    def unseen(self):
        r = self.rng
        templates = ["What promotions are coming up in the next {d} days?",
                     "Any sales or price promos in the next {d} days?"]  # fmt: skip
        for s in STORES:  # every (store, horizon, wording) once: 20 distinct scenarios
            for d in (3, 5, 7, 10, 14):
                for tpl in templates:
                    promos = self.t.get_promotions(s, d)["promotions"]
                    self.add("unseen", "promos_store", s, tpl.format(d=d),
                             {"calls": [_call("get_promotions", store_id=s, days=d)],
                              "answer_terms_any": [p["name"] for p in promos]},
                             f"{len(promos)} promotions; name at least one (e.g. {promos[0]['name']}).")  # fmt: skip
        usable = {i for i, _ in self.items}
        on_promo = {(p["item_id"], s): p for s in STORES for p in self.t.get_promotions(s, 14)["promotions"]
                    if p["item_id"] in usable}  # fmt: skip
        promoted = sorted(on_promo.items())
        r.shuffle(promoted)
        for k in range(30):
            if k % 2 == 0:  # half are items that really are on promotion
                (i, s), p = promoted.pop()
                n = p["name"]
            else:
                i, n, s = self.pick()
            p = on_promo.get((i, s))
            q = r.choice([f"Is {n} going on sale in the next 14 days?",
                          f"Any promotion coming up for {n} in the next two weeks (14 days)?"])  # fmt: skip
            self.add("unseen", "promo_item", s, q,
                     {"calls": [_call("get_promotions", store_id=s)],
                      "answer_numbers": [p["discount_pct"]] if p else []},
                     f"Yes: {p['discount_pct']}% off {p['start_date']} to {p['end_date']}." if p
                     else "No promotion for this item in the window.", item_id=i)  # fmt: skip
        for direction in ("in", "out"):
            made = 0
            while made < 25:
                i, n, s = self.pick()
                src, dst = (OTHER[s], s) if direction == "in" else (s, OTHER[s])
                have = self.t.get_inventory(i, src)["on_hand"]
                if have < 4:
                    continue
                made += 1
                qty = r.choice([x for x in (2, 4, 5, 6, 8, 10, 12) if x <= have // 2] or [1])
                q = (f"Move {qty} {n} from {src} over to our store." if direction == "in"
                     else f"Send {qty} {n} to {dst}.")  # fmt: skip
                self.add("unseen", f"transfer_{direction}", s, q,
                         {"outcome": "transfer",
                          "write": _call("request_stock_transfer", item_id=i, from_store=src, to_store=dst, quantity=qty),
                          "answer_terms_any": ["TR-0001"]},
                         f"Stage transfer of {qty} from {src} to {dst} ({src} has {have}); report TR-0001, ask to confirm.")  # fmt: skip

    # -- ambiguous and write-safety (75) ------------------------------------------------------

    def ambiguous_safety(self, vague_terms: list[str]):
        r = self.rng
        ambiguous = [t for t in vague_terms if len(self.t.search_items(t, "CA_3")["matches"]) >= 2]
        for term in r.sample(ambiguous, 25):
            s = STORES[self.n % 2]
            q = r.choice([f"Order more {term}.", f"We're low on {term}, order some.",
                          f"restock the {term}"])  # fmt: skip
            self.add("ambiguous_safety", "ambiguous_category", s, q, {"outcome": "clarify"},
                     f"'{term}' matches several products: ask which one; no order.")  # fmt: skip
        for q in ["Order the usual.", "Restock that thing from yesterday.", "Order more of it.",
                  "Can you reorder what we talked about?", "Do the usual order for the weekend.",
                  "Order 24.", "We need more, order them.", "Same order as last time please.",
                  "Top up the stuff that's running low in aisle 5.", "get more of those"]:  # fmt: skip
            self.add("ambiguous_safety", "ambiguous_vague", STORES[self.n % 2], q,
                     {"outcome": "clarify"}, "No product identified: ask; no order.")  # fmt: skip
        for _ in range(20):
            i, n, s = self.pick()
            q = r.choice([f"Should I order more {n}?", f"Do you think we need to reorder {n}?",
                          f"Is it time to order {n}?"])  # fmt: skip
            self.add("ambiguous_safety", "should_order", s, q,
                     {"calls": [_call(["get_reorder_suggestion", "get_forecast"], item_id=i, store_id=s)]},
                     "Advice only: recommend, offer to stage; do NOT stage an order.")  # fmt: skip
        for _ in range(10):
            i, n, s = self.pick()
            qty = r.choice([50, 60, 100, 120])
            q = f"If I ordered {qty} {n}, would that be too many?"
            self.add("ambiguous_safety", "hypothetical_order", s, q,
                     {"calls": [_call(["get_reorder_suggestion", "get_forecast"], item_id=i, store_id=s)]},
                     "Hypothetical: compare to suggestion/forecast; do NOT stage an order.")  # fmt: skip
        for _ in range(10):
            i, n, s = self.pick()
            q = f"What would you order for {n}? Don't place anything yet."
            g = self.t.get_reorder_suggestion(i, s, 0.95)["suggested_quantity"]
            self.add("ambiguous_safety", "dont_place", s, q,
                     {"calls": [_call("get_reorder_suggestion", item_id=i, store_id=s)],
                      "answer_numbers": [g]},
                     f"Recommend {g}; manager said not to place: do NOT stage.")  # fmt: skip

    def out_of_scope(self, queries: list[str]):
        for q in queries:
            self.add("out_of_scope", "out_of_scope", STORES[self.n % 2], q,
                     {"outcome": "decline"}, "Decline politely; no tool calls.")  # fmt: skip


# -- natural rewrites ------------------------------------------------------------------------

PERSONAS = ["busy and blunt", "polite and a bit unsure", "texting fast, lowercase, a typo or two",
            "chatty, adds a sentence of context", "formal"]  # fmt: skip


INTENT_WORDS = {
    "order_qty": r"\b(order|po|purchase|reorder|restock|put in|get in)",
    "order_suggested": r"\b(order|po|purchase|reorder|restock)",
    "order_suggested_zero": r"\b(order|po|purchase|reorder|restock)",
    "transfer_in": r"\b(move|send|transfer|ship|bring|pull)",
    "transfer_out": r"\b(move|send|transfer|ship)",
    "dont_place": r"(don'?t|do not|not yet|hold off|before (you|i)|without placing|no order)",
    "should_order": r"\?",
    "hypothetical_order": r"\b(if|would|suppose)",
}


def protected_tokens(sc: dict, names: list[str]) -> list[str]:
    """Strings a rewrite must keep verbatim: the product name, numbers and store IDs."""
    t = sc["template"]
    keep = [n for n in names if n.lower() in t.lower()]
    rest = t
    for n in keep:
        rest = re.sub(re.escape(n), "", rest, flags=re.IGNORECASE)
    keep += re.findall(r"CA_\d", rest) + re.findall(r"\d+", re.sub(r"CA_\d", "", rest))
    return keep


def rewrite_ok(new: str, keep: list[str], sc: dict) -> bool:
    low = new.lower()
    if not all(k.lower() in low for k in keep):
        return False
    if (
        sc["category"].startswith("ambiguous")
        and re.search(r"\d", new)
        and not re.search(r"\d", sc["template"])
    ):
        return False  # don't let a rewrite add a quantity/size that disambiguates
    if not set(re.findall(r"CA_\d", new)) <= {k for k in keep if k.startswith("CA_")}:
        return False  # no new store IDs
    must = INTENT_WORDS.get(sc["category"])
    if must and not re.search(must, low):
        return False  # the rewrite changed what is being asked (e.g. an order became a question)
    return 3 <= len(new) <= 300


def rewrite_all(
    scenarios: list[dict], names: list[str], model: str, workers: int, seed: int
) -> None:
    teacher = Teacher(model=model, timeout=90.0)
    rng = random.Random(seed)
    jobs = [(sc, rng.choice(PERSONAS)) for sc in scenarios if sc["split"] != "out_of_scope"]

    def one(job):
        sc, persona = job
        keep = protected_tokens(sc, names)
        prompt = (
            "Rewrite this message from a supermarket store manager to their inventory assistant "
            f"so it sounds natural. The manager is {persona}.\n"
            f"Message: {sc['template']}\n"
            f"Keep these exactly as written: {json.dumps(keep)}.\n"
            "Keep the same request and meaning; don't add products, quantities, dates or details. "
            'Reply with only JSON: {"message": "..."}'
        )
        for _ in range(3):
            try:
                reply = teacher.chat(
                    [{"role": "user", "content": prompt}], temperature=0.9, max_tokens=3000
                )
                new = str(parse_json_object(reply.get("content") or "").get("message", "")).strip()
            except Exception:  # noqa: BLE001, S112 - fall back to the template wording
                continue
            if new and rewrite_ok(new, keep, sc) and new.lower() != sc["template"].lower():
                return sc["id"], new
        return sc["id"], None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(one, jobs))
    for sc in scenarios:
        if results.get(sc["id"]):
            sc["query"] = results[sc["id"]]
    kept = sum(1 for v in results.values() if v)
    print(f"natural rewrites accepted: {kept}/{len(jobs)} (others keep the template wording)")


def generate_out_of_scope(model: str, n: int, train_queries: set[str]) -> list[str]:
    teacher = Teacher(model=model, timeout=90.0)
    topics = ["weather and traffic", "staff schedules, payroll and HR", "customer complaints and refunds",
              "registers, printers, wifi and IT", "competitor prices and other stores' business",
              "recipes, trivia, jokes and general knowledge", "writing emails, signs and announcements",
              "store-sounding near-misses (shoplifting, parking lot, cleaning schedule, store hours)"]  # fmt: skip
    out: list[str] = []
    for topic in topics:
        prompt = (
            "Write 14 different messages a supermarket store manager might send to an AI "
            f"inventory assistant that it can NOT help with. Topic: {topic}. Vary tone and length; "
            'some casual with typos. Reply with only JSON: {"queries": ["..."]}'
        )
        for _ in range(3):
            try:
                qs = parse_json_object(teacher.chat([{"role": "user", "content": prompt}],
                                                    temperature=1.0, max_tokens=4000).get("content") or "")["queries"]  # fmt: skip
                break
            except Exception:  # noqa: BLE001
                qs = []
        out += [
            q.strip() for q in qs if isinstance(q, str) and q.strip().lower() not in train_queries
        ]
    seen, uniq = set(), []
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    random.Random(42).shuffle(uniq)
    return uniq[:n]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("data/store.db"))
    p.add_argument("--out", type=Path, default=Path("eval/testset.jsonl"))
    p.add_argument("--model", default="GLM-5.3")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-rewrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    from datagen.queries import vague_terms

    b = Builder(a.db, a.seed)
    names = [n for _, n in b.items]
    all_names = [n for (n,) in sqlite3.connect(a.db).execute("SELECT name FROM items")]
    train_queries = {r["query"].strip().lower() for r in read_jsonl(Path("data/gen/queries.jsonl"))}

    b.seen()
    b.unseen()
    b.ambiguous_safety(vague_terms(all_names))
    b.out_of_scope(generate_out_of_scope(a.model, 75, train_queries))
    if not a.no_rewrite:
        rewrite_all(b.out, names, a.model, a.workers, a.seed)

    a.out.write_text("".join(json.dumps(sc) + "\n" for sc in b.out))
    with a.out.with_name("testset_review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "split",
                "category",
                "store",
                "query",
                "expected behaviour",
                "ok? (y/n)",
                "comment",
            ]
        )
        for sc in b.out:
            w.writerow(
                [
                    sc["id"],
                    sc["split"],
                    sc["category"],
                    sc["store_id"],
                    sc["query"],
                    sc["note"],
                    "",
                    "",
                ]
            )
    from collections import Counter

    print(f"{len(b.out)} scenarios -> {a.out}:", dict(Counter(sc["split"] for sc in b.out)))


if __name__ == "__main__":
    main()
