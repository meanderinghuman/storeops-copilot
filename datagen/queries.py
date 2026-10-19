"""Step 2.1: generate store-manager questions across the five request kinds with the teacher.

    uv run python -m datagen.queries --model DeepSeek-V4.1-Flash

Each batch asks for 10 questions of one kind, seeded with real catalog item names, a persona and
a writing style, so the set is varied. Only training stores are used (held-out stores are kept
for the test set). Output: data/gen/queries.jsonl with {id, kind, store_id, query, seed_items}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datagen.runner import append_jsonl, read_jsonl
from datagen.teacher import Teacher, parse_json_object

TARGETS = {"lookup": 550, "planning": 650, "action": 600, "ambiguous": 350, "out_of_scope": 350}
BATCH = 10

KIND_INSTRUCTIONS = {
    "lookup": (
        "Each question asks for one simple fact about one specific product from the list: units on "
        "hand, units on order, pack size, supplier lead time, shelf price, or units sold in recent "
        "weeks. Name the product clearly enough to identify it (you may shorten the name)."
    ),
    "planning": (
        "Each question needs a judgement about one specific product from the list: will we run out "
        "before a day or the weekend, is stock enough for the next N days, how much will we sell, "
        "is demand going up or down, should we reorder or how many should we order. The manager "
        "is asking for advice only and does NOT ask you to place an order. Name the product "
        "clearly enough to identify it."
    ),
    "action": (
        "Each request explicitly tells the assistant to place, stage or create a purchase order "
        "for one specific product from the list. About half give a quantity in units (e.g. 'order 24'), the "
        "rest ask the assistant to work out how many (e.g. 'enough for next week'). Name the "
        "product clearly enough to identify it."
    ),
    "ambiguous": (
        "Each request is unclear, so a careful assistant must ask a clarifying question before "
        "doing anything: e.g. 'order the usual', 'do we have enough of that stuff', a vague "
        "category that matches several products (use the category words given), 'order more' "
        "with no product, or a pronoun with nothing to refer to. Don't name a specific product."
    ),
    "out_of_scope": (
        "Each request is something an inventory assistant can't help with: weather, staff "
        "schedules, payroll, customer complaints, register or IT problems, competitor prices, "
        "recipes, general knowledge, writing emails, jokes, news. Include some near-misses that "
        "sound store-related but aren't about this store's inventory, sales or ordering."
    ),
}

PERSONAS = [
    "a new assistant store manager, polite and a bit unsure",
    "a veteran store manager who is blunt and busy",
    "a manager texting from the sales floor, very terse",
    "a manager whose first language isn't English",
    "a department lead covering for the store manager",
    "a night-shift manager doing end-of-day checks",
]
STYLES = [
    "mostly short questions (3-10 words)",
    "a mix of short and long questions, some with a sentence of context first",
    "casual wording, lowercase, a few typos and abbreviations",
    "full sentences, slightly formal",
    "varied: some questions, some commands, different openings",
]

_STOP = {"with", "pack", "count", "inch", "gallon", "liter", "roll", "rolls", "free", "large", "heavy",
         "duty", "stainless", "steel", "kitchen", "plastic", "glass", "silicone", "organic", "fresh",
         "multi", "purpose", "extra", "light", "mild", "ultra", "set", "absorbent", "black",
         "booster", "brown", "bunch", "collapsible", "concentrate", "disposable", "reusable", "grain",
         "liquid", "quick", "whole", "covers", "holder", "drying", "cutting", "mixing", "canned"}  # fmt: skip


def vague_terms(names: list[str]) -> list[str]:
    """Category words shared by 3+ products (e.g. 'sponges', 'rice'): naturally ambiguous."""
    counts = Counter(
        w for n in names for w in set(re.findall(r"[a-z]{4,}", n.lower())) if w not in _STOP
    )
    return sorted(w for w, c in counts.items() if c >= 3)


def build_prompt(kind: str, items: list[str], vague: list[str], persona: str, style: str) -> str:
    seed = (
        f"Category words that match several products: {', '.join(vague)}"
        if kind == "ambiguous"
        else f"Products in this store: {'; '.join(items)}"
    )
    return (
        "You are writing realistic messages that a supermarket store manager would type to an AI "
        "inventory assistant for their own store.\n"
        f"Write {BATCH} different messages.\n"
        f"Kind: {KIND_INSTRUCTIONS[kind]}\n"
        f"{seed}\n"
        f"The manager is {persona}. Style: {style}.\n"
        "Don't mention store IDs, dates or the assistant's name. Don't number the messages.\n"
        'Reply with only a JSON object: {"queries": ["...", "..."]}'
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="defaults to TEACHER_MODEL")
    p.add_argument("--db", type=Path, default=Path("data/store.db"))
    p.add_argument("--out", type=Path, default=Path("data/gen/queries.jsonl"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--scale", type=float, default=1.0, help="multiply per-kind targets")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    conn = sqlite3.connect(a.db)
    names = [n for (n,) in conn.execute("SELECT name FROM items ORDER BY item_id")]
    stores = [
        s for (s,) in conn.execute("SELECT store_id FROM stores WHERE split='train' ORDER BY 1")
    ]
    conn.close()
    vague = vague_terms(names)
    teacher = Teacher(model=a.model, timeout=90.0)
    rng = random.Random(a.seed)

    existing = read_jsonl(a.out)
    have = Counter(r["kind"] for r in existing)
    seen = {r["query"].strip().lower() for r in existing}
    jobs = []
    for kind, target in TARGETS.items():
        need = max(0, round(target * a.scale) - have[kind])
        for _ in range(-(-need // BATCH)):
            items = rng.sample(names, 8)
            terms = rng.sample(vague, min(8, len(vague)))
            jobs.append(
                (
                    kind,
                    items,
                    build_prompt(kind, items, terms, rng.choice(PERSONAS), rng.choice(STYLES)),
                )
            )
    print(f"{len(existing)} queries on disk; {len(jobs)} batches to generate with {teacher.model}")

    def one(job):
        kind, items, prompt = job
        for attempt in range(3):  # reasoning models sometimes return no content; just retry
            try:
                reply = teacher.chat(
                    [{"role": "user", "content": prompt}], temperature=1.0, max_tokens=6000
                )
                qs = parse_json_object(reply.get("content") or "").get("queries", [])
                break
            except Exception as exc:  # noqa: BLE001 - a failed batch is just skipped
                if attempt == 2:
                    print(f"  batch failed ({kind}): {exc}")
                    return kind, items, []
        return (
            kind,
            items,
            [q.strip() for q in qs if isinstance(q, str) and 2 <= len(q.strip()) <= 400],
        )

    n = len(existing)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for kind, items, qs in pool.map(one, jobs):
            for q in qs:
                if q.lower() in seen:
                    continue
                seen.add(q.lower())
                rec = {"id": f"q{n:05d}", "kind": kind, "store_id": rng.choice(stores), "query": q,
                       "seed_items": items if kind in ("lookup", "planning", "action") else []}  # fmt: skip
                append_jsonl(a.out, rec)
                n += 1
    counts = Counter(r["kind"] for r in read_jsonl(a.out))
    print(f"{sum(counts.values())} queries: {json.dumps(dict(counts))}")


if __name__ == "__main__":
    main()
