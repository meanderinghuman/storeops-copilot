"""Build the 100-scenario "hard" split (eval/testset_hard.jsonl), kept separate from the main 400.

    uv run python -m eval.build_hard

The main test set is single-intent and clean; a large API model scores 100% on it. This split adds
headroom: misspelled product names, two products in one message, follow-up turns that depend on
earlier context, and tricky requests (negated orders, stores that don't exist, products the store
doesn't carry). Same held-out stores (CA_3 / CA_4) and the same scenario format; follow-ups carry
the earlier turns in "history" (built from real tool results).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from eval.build_testset import Builder, _call
from store.tools import result_json

NEGATIVE = ["n't", "not ", "no ", "unable", "unknown", "invalid"]
NOT_CARRIED = ["unicorn cupcake toppers", "snow tires", "goldfish food flakes", "printer ink cartridges",
               "phone chargers", "dog leashes", "motor oil 5 qt", "baby formula 12 oz", "lottery tickets",
               "fresh salmon fillets", "propane tanks", "car air fresheners"]  # fmt: skip


class HardBuilder(Builder):
    def add_hard(self, category, store_id, query, expected, note, **extra):
        self.add(
            "hard", category, store_id, query, expected, note, item_id=extra.pop("item_id", None)
        )
        sc = self.out[-1]
        sc["id"] = "h" + sc["id"][1:]
        sc.update(extra)

    def typo(self, item_id: str, name: str, store: str) -> str | None:
        """A misspelling of the name that our search still resolves to this item first."""
        for _ in range(20):
            words = name.split()
            spots = [k for k, w in enumerate(words) if re.fullmatch(r"[A-Za-z-]{4,}", w)]
            if not spots:
                return None
            k = self.rng.choice(spots)
            w, j = words[k], self.rng.randrange(1, len(words[k]) - 1)
            words[k] = self.rng.choice(
                [w[:j] + w[j + 1 :], w[:j] + w[j + 1] + w[j] + w[j + 2 :], w[:j] + w[j] + w[j:]]
            )
            text = " ".join(words)
            if self.rng.random() < 0.5:
                text = re.sub(r"(\d) (oz|lb|ft|in)\b", r"\1\2", text).lower()
            if text.lower() == name.lower():
                continue
            hits = self.t.search_items(text, store)["matches"]
            if hits and hits[0]["item_id"] == item_id:
                return text
        return None

    def typos(self):
        made = Counter()
        while sum(made.values()) < 25:
            i, n, s = self.pick()
            t = self.typo(i, n, s)
            if not t:
                continue
            if made["lookup"] < 10:
                v = self.t.get_inventory(i, s)
                self.add_hard("typo_lookup", s, self.rng.choice([f"how many {t} we got", f"stock on {t}?"]),
                              {"calls": [_call("get_inventory", item_id=i, store_id=s)], "answer_numbers": [v["on_hand"]]},
                              f"Misspelled '{n}': find it, report on hand = {v['on_hand']}.", item_id=i)  # fmt: skip
                made["lookup"] += 1
            elif made["order"] < 8:
                qty = self.rng.choice([6, 12, 24])
                self.add_hard("typo_order", s, f"order {qty} {t}",
                              {"outcome": "order", "answer_terms_any": ["PO-0001"],
                               "write": _call("create_purchase_order", item_id=i, store_id=s, quantity=qty)},
                              f"Misspelled '{n}': stage PO for {qty}, ask to confirm.", item_id=i)  # fmt: skip
                made["order"] += 1
            else:
                d = self.rng.choice([5, 7])
                f = self.t.get_forecast(i, s, d)["total"]
                self.add_hard("typo_forecast", s, f"how many {t} will we sell next {d} days",
                              {"calls": [_call("get_forecast", item_id=i, store_id=s, days=d)], "answer_numbers": [f["p50"]]},
                              f"Misspelled '{n}': forecast p50 = {f['p50']} over {d} days.", item_id=i)  # fmt: skip
                made["forecast"] += 1

    def two_items(self):
        r = self.rng
        for k in range(25):
            (a, na, s), (b, nb, _) = self.pick(), self.pick()
            if k < 10:
                va, vb = self.t.get_inventory(a, s), self.t.get_inventory(b, s)
                q = r.choice(
                    [f"How many {na} and {nb} do we have?", f"stock check: {na}, and also {nb}"]
                )
                self.add_hard("two_lookup", s, q,
                              {"calls": [_call("get_inventory", item_id=a, store_id=s), _call("get_inventory", item_id=b, store_id=s)],
                               "answer_numbers": [va["on_hand"], vb["on_hand"]]},
                              f"Two lookups: {va['on_hand']} and {vb['on_hand']} on hand.", item_ids=[a, b])  # fmt: skip
            elif k < 18:
                qa, qb = r.choice([6, 12]), r.choice([18, 24])
                self.add_hard("two_order", s, f"Order {qa} {na} and {qb} {nb}.",
                              {"outcome": "order", "answer_terms_all": ["PO-0001", "PO-0002"],
                               "writes": [_call("create_purchase_order", item_id=a, store_id=s, quantity=qa),
                                          _call("create_purchase_order", item_id=b, store_id=s, quantity=qb)]},
                              f"Two POs: {qa} and {qb}; report PO-0001 and PO-0002, ask to confirm.", item_ids=[a, b])  # fmt: skip
            else:
                ta, tb = (self.t.get_sales_history(x, s, 4)["total_units"] for x in (a, b))
                if ta == tb:
                    continue
                self.add_hard("compare_two", s, f"Which sold more over the last 4 weeks: {na} or {nb}?",
                              {"calls": [_call("get_sales_history", item_id=a, store_id=s, weeks=4),
                                         _call("get_sales_history", item_id=b, store_id=s, weeks=4)],
                               "answer_numbers": [ta, tb]},
                              f"{na if ta > tb else nb} sold more ({max(ta, tb)} vs {min(ta, tb)}).", item_ids=[a, b])  # fmt: skip

    # -- follow-ups ---------------------------------------------------------------------------

    def _turns(self, question: str, steps: list[tuple[str, dict, dict]], answer: str) -> list[dict]:
        msgs = [{"role": "user", "content": question}]
        for k, (name, args, result) in enumerate(steps):
            cid = f"call_h{k}"
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})  # fmt: skip
            msgs.append(
                {"role": "tool", "tool_call_id": cid, "name": name, "content": result_json(result)}
            )
        msgs.append({"role": "assistant", "content": answer})
        return msgs

    def should_order_history(self, i, n, s):
        g = self.t.get_reorder_suggestion(i, s, 0.95)
        hist = self._turns(
            f"Should I order more {n}?",
            [("search_items", {"query": n, "store_id": s}, self.t.search_items(n, s)),
             ("get_reorder_suggestion", {"item_id": i, "store_id": s}, g)],
            f"I'd order {g['suggested_quantity']} units. You have {g['on_hand']} on hand and {g['on_order']} on "
            f"order, and the suggestion at a 95% service level is {g['suggested_quantity']}. "
            f"Want me to stage a purchase order for {g['suggested_quantity']}?",
        )  # fmt: skip
        return g["suggested_quantity"], hist

    def followups(self):
        r = self.rng
        made = Counter()
        while sum(made.values()) < 25:
            i, n, s = self.pick()
            if made["confirm"] + made["change"] + made["decline"] < 20:
                g, hist = self.should_order_history(i, n, s)
                if g == 0:
                    continue
                if made["confirm"] < 10:
                    self.add_hard("followup_confirm", s, r.choice(["yes, go ahead", "Yes please, stage it.", "ok do it"]),
                                  {"outcome": "order", "answer_terms_any": ["PO-0001"],
                                   "write": _call("create_purchase_order", item_id=i, store_id=s, quantity=g)},
                                  f"Confirmed the offer: stage PO for {g}, ask to confirm.", item_id=i, history=hist)  # fmt: skip
                    made["confirm"] += 1
                elif made["change"] < 5:
                    q2 = r.choice([x for x in (6, 18, 24, 36, 48) if x != g])
                    self.add_hard("followup_change", s, r.choice([f"make it {q2} instead", f"Let's do {q2}, not {g}."]),
                                  {"outcome": "order", "answer_terms_any": ["PO-0001"],
                                   "write": _call("create_purchase_order", item_id=i, store_id=s, quantity=q2)},
                                  f"Changed quantity: stage PO for {q2} (not {g}), ask to confirm.", item_id=i, history=hist)  # fmt: skip
                    made["change"] += 1
                else:
                    self.add_hard("followup_decline", s, r.choice(["no, hold off for now", "Not yet, thanks."]),
                                  {}, "Manager declined: no order, no further tool calls needed.", history=hist)  # fmt: skip
                    made["decline"] += 1
            else:
                b, nb, _ = self.pick()
                va, vb = self.t.get_inventory(i, s), self.t.get_inventory(b, s)
                hist = self._turns(
                    f"How many {n} do we have?",
                    [("search_items", {"query": n, "store_id": s}, self.t.search_items(n, s)),
                     ("get_inventory", {"item_id": i, "store_id": s}, va)],
                    f"You have {va['on_hand']} {n} on hand.",
                )  # fmt: skip
                self.add_hard("followup_other", s, f"and what about {nb}?",
                              {"calls": [_call("get_inventory", item_id=b, store_id=s)], "answer_numbers": [vb["on_hand"]]},
                              f"Same question for {nb}: on hand = {vb['on_hand']}.", item_id=b, history=hist)  # fmt: skip
                made["other"] += 1

    # -- tricky ---------------------------------------------------------------------------------

    def tricky(self):
        r = self.rng
        for _ in range(9):
            i, n, s = self.pick()
            g = self.t.get_reorder_suggestion(i, s, 0.95)["suggested_quantity"]
            q = r.choice([f"Don't order {n}, just tell me what the suggestion would be.",
                          f"Hold off on ordering {n}, but how many would you suggest?",
                          f"No order for {n} yet. What's the suggested quantity?"])  # fmt: skip
            self.add_hard("negated_order", s, q,
                          {"calls": [_call("get_reorder_suggestion", item_id=i, store_id=s)], "answer_numbers": [g]},
                          f"Report suggestion {g}; manager said not to order: do NOT stage.")  # fmt: skip
        for k in range(8):
            i, n, s = self.pick()
            bad = ["CA_9", "CA_7", "TX_1", "CA_12"][k % 4]
            q = r.choice([f"How many {n} does {bad} have?", f"Check stock of {n} at {bad}."])
            self.add_hard("unknown_store", s, q,
                          {"answer_terms_all": [bad], "answer_terms_any": NEGATIVE},
                          f"{bad} isn't a store in the system: say so; don't invent numbers or order.", item_id=i)  # fmt: skip
        made = 0
        for thing in NOT_CARRIED:
            s = ("CA_3", "CA_4")[made % 2]
            if made == 8 or self.t.search_items(thing, s)["matches"]:
                continue
            q = r.choice([f"How many {thing} do we have?", f"Order 12 {thing}."])
            self.add_hard("not_carried", s, q,
                          {"calls": [_call("search_items", store_id=s)], "answer_terms_any": NEGATIVE},
                          "Search finds nothing: say the store doesn't carry it; no order, no invented numbers.")  # fmt: skip
            made += 1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("data/store.db"))
    p.add_argument("--out", type=Path, default=Path("eval/testset_hard.jsonl"))
    p.add_argument("--seed", type=int, default=43)
    a = p.parse_args(argv)

    b = HardBuilder(a.db, a.seed)
    b.typos()
    b.two_items()
    b.followups()
    b.tricky()
    for k, sc in enumerate(b.out, start=1):  # renumber after skipped attempts
        sc["id"] = f"h{k:03d}"
        sc["template"] = sc["query"]
    a.out.write_text("".join(json.dumps(sc) + "\n" for sc in b.out))
    with a.out.with_name("testset_hard_review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "category",
                "store",
                "earlier turns",
                "query",
                "expected behaviour",
                "ok? (y/n)",
                "comment",
            ]
        )
        for sc in b.out:
            prior = " | ".join(
                m["content"]
                for m in sc.get("history", [])
                if m["role"] in ("user", "assistant") and m["content"]
            )
            w.writerow(
                [sc["id"], sc["category"], sc["store_id"], prior, sc["query"], sc["note"], "", ""]
            )
    print(f"{len(b.out)} hard scenarios -> {a.out}:", dict(Counter(sc["category"] for sc in b.out)))


if __name__ == "__main__":
    main()
