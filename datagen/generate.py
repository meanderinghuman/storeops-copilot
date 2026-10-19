"""Steps 2.2-2.4: run the teacher as an agent on every query, filter, and report retention.

    uv run python -m datagen.generate --model gpt-oss-120b

Resumable: queries already in conversations.jsonl are skipped. Outputs in data/gen/:
    conversations.jsonl  every run (kept or not) with its rejection reasons
    verified.jsonl       kept conversations as {"id", "kind", "messages", "tools"}
    stats.json           retention overall and per kind, top rejection reasons
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from datagen.filters import KINDS
from datagen.runner import append_jsonl, read_jsonl, run_many
from store.tools import tool_specs


def to_training_messages(messages: list[dict]) -> list[dict]:
    """Chat-template-ready messages: tool-call arguments as objects (Qwen's template serialises
    them itself), no API-only fields on tool results."""
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k != "tool_call_id"}
        if m.get("tool_calls"):
            m["content"] = ""  # drop occasional filler like "I'll look that up." (~7% of turns)
            m["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"]),
                    },
                }
                for tc in m["tool_calls"]
            ]
        out.append(m)
    return out


def latest_per_query(records: list[dict]) -> list[dict]:
    """A query may have been retried after an API error; keep its most recent record."""
    return list({r["id"]: r for r in records}.values())


def write_outputs(records: list[dict], out_dir: Path) -> dict:
    records = latest_per_query(records)
    kept = [r for r in records if not r["reasons"]]
    with (out_dir / "verified.jsonl").open("w") as f:
        for r in sorted(kept, key=lambda r: r["id"]):
            row = {"id": r["id"], "kind": r["kind"], "store_id": r["store_id"],
                   "messages": to_training_messages(r["conversation"]["messages"]),
                   "tools": tool_specs()}  # fmt: skip
            f.write(json.dumps(row) + "\n")
    total = Counter(r["kind"] for r in records)
    kept_by = Counter(r["kind"] for r in kept)
    stats = {
        "model": records[0]["model"] if records else None,
        "conversations": len(records),
        "kept": len(kept),
        "retention": round(len(kept) / max(len(records), 1), 3),
        "by_kind": {
            k: {
                "total": total[k],
                "kept": kept_by[k],
                "retention": round(kept_by[k] / max(total[k], 1), 3),
            }
            for k in KINDS
        },
        "rejection_reasons": dict(Counter(x for r in records for x in r["reasons"]).most_common()),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--db", type=Path, default=Path("data/store.db"))
    p.add_argument("--queries", type=Path, default=Path("data/gen/queries.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/gen"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="only the first N queries")
    a = p.parse_args(argv)

    conv_path = a.out_dir / "conversations.jsonl"
    done = {r["id"] for r in latest_per_query(read_jsonl(conv_path)) if "api_error" not in r}
    queries = read_jsonl(a.queries)[: a.limit]
    todo = [q for q in queries if q["id"] not in done]
    print(f"{len(queries)} queries, {len(done)} already done, {len(todo)} to run with {a.model}")

    start, kept, api_errors_in_a_row = time.monotonic(), 0, 0
    for n, rec in enumerate(
        run_many(todo, a.model, a.db, a.out_dir / "scratch", workers=a.workers), 1
    ):
        append_jsonl(conv_path, rec)
        kept += not rec["reasons"]
        api_errors_in_a_row = api_errors_in_a_row + 1 if "api_error" in rec else 0
        if api_errors_in_a_row >= 20:
            print(f"stopping: 20 API errors in a row (VPN down?). Last: {rec['api_error'][:200]}")
            print("Re-run the same command to resume; failed queries are retried.")
            break
        if n % 25 == 0 or n == len(todo):
            rate = n / (time.monotonic() - start)
            print(f"  {n}/{len(todo)}  kept {kept / n:.0%}  {rate * 60:.0f}/min", flush=True)

    stats = write_outputs(read_jsonl(conv_path), a.out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
