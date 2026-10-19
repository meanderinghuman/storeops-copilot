"""Pick the teacher: run a fixed set of 20 queries (all five request kinds) through each candidate
model with real tool execution and compare filter pass rates, failure reasons and speed.

    uv run python -m datagen.bakeoff --models gpt-oss-120b GLM-5.3 DeepSeek-V4.1-Flash
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from datagen.runner import append_jsonl, run_many

QUERIES = [
    ("lookup", "CA_1", "How many Paper towels 6-pack do we have?"),
    ("lookup", "CA_2", "whats our stock on toilet paper"),
    ("lookup", "CA_1", "How many red grapes did we sell over the last 2 weeks?"),
    ("lookup", "CA_2", "price of the jasmine rice?"),
    ("planning", "CA_1", "Will we run out of bananas before the weekend?"),
    ("planning", "CA_2", "Is the olive oil going to last through next week?"),
    ("planning", "CA_1", "How many rolled oats do you expect us to sell in the next 5 days?"),
    ("planning", "CA_1", "Should I order more toilet paper for this weekend?"),
    ("planning", "CA_2", "the nonstick frying pan has been slow lately, do we have too many?"),
    ("action", "CA_1", "Order enough Paper towels 6-pack for next week."),
    ("action", "CA_2", "Please place an order for 24 Dish soap liquid 20 oz."),
    ("action", "CA_1", "go ahead and reorder the red seedless grapes"),
    ("action", "CA_2", "We're low on the Long grain white rice 5 lb, order what we need."),
    ("action", "CA_1", "Stage a PO for the heavy duty trash bags, whatever the system suggests"),
    ("ambiguous", "CA_1", "Order more of the usual."),
    ("ambiguous", "CA_2", "Order more paper towels."),
    ("ambiguous", "CA_1", "Do we have enough sponges?"),
    ("out_of_scope", "CA_1", "What's the weather tomorrow?"),
    ("out_of_scope", "CA_2", "Can you write me a cover letter for a shift lead job?"),
    ("out_of_scope", "CA_1", "What's Walmart's stock price today?"),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+", default=["gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash"]
    )
    p.add_argument("--db", type=Path, default=Path("data/store.db"))
    p.add_argument("--out", type=Path, default=Path("data/gen/bakeoff"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)

    tasks = [
        {"id": f"b{i:02d}", "kind": k, "store_id": s, "query": q}
        for i, (k, s, q) in enumerate(QUERIES)
    ]
    summary = []
    for model in a.models:
        out = a.out / f"{model}.jsonl"
        out.unlink(missing_ok=True)
        recs = []
        for rec in run_many(tasks, model, a.db, a.out / "scratch", workers=a.workers):
            append_jsonl(out, rec)
            recs.append(rec)
            status = "keep" if not rec["reasons"] else ",".join(rec["reasons"])
            print(f"  {model:22s} {rec['id']} {rec['kind']:12s} {status}", flush=True)
        secs = [r["conversation"]["seconds"] for r in recs if "conversation" in r]
        kept = sum(not r["reasons"] for r in recs)
        by_kind = Counter(r["kind"] for r in recs if not r["reasons"])
        reasons = Counter(x for r in recs for x in r["reasons"])
        summary.append((model, kept, by_kind, reasons, statistics.median(secs) if secs else 0))

    print("\nmodel                  kept   median s  kept by kind / rejection reasons")
    for model, kept, by_kind, reasons, med in summary:
        print(f"{model:22s} {kept:2d}/{len(tasks)}  {med:6.1f}    {dict(by_kind)}  {dict(reasons)}")


if __name__ == "__main__":
    main()
