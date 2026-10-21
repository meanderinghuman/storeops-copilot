"""Load test: replay test-set questions against the gateway at increasing concurrency.

    uv run python -m loadtest.loadtest --label metal-2slots --users 1 2 4 8 16 --seconds 120

Each virtual user sends a question, waits for the full answer (all tool rounds), and immediately
sends the next. Measures per-conversation latency (p50/p95), conversations/s and errors at each
concurrency level. Point it at a gateway with escalation OFF to measure local capacity.
Results: eval/results/loadtest_<label>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else 0.0


async def level(url: str, queries: list[dict], users: int, seconds: float, timeout: float) -> dict:
    lat, errors, routes = [], 0, {}
    deadline = time.monotonic() + seconds
    rng = random.Random(users)

    async def user(client: httpx.AsyncClient) -> None:
        nonlocal errors
        while time.monotonic() < deadline:
            q = rng.choice(queries)
            t0 = time.monotonic()
            try:
                r = await client.post(url, json={"store_id": q["store_id"], "message": q["query"]})
                r.raise_for_status()
                routes[r.json()["route"]] = routes.get(r.json()["route"], 0) + 1
                lat.append(time.monotonic() - t0)
            except (httpx.HTTPError, KeyError, ValueError):
                errors += 1

    start = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        await asyncio.gather(*(user(client) for _ in range(users)))
    wall = time.monotonic() - start
    return {
        "users": users,
        "conversations": len(lat),
        "errors": errors,
        "conv_per_s": round(len(lat) / wall, 3),
        "p50_s": pct(lat, 0.5),
        "p95_s": pct(lat, 0.95),
        "mean_s": round(statistics.mean(lat), 2) if lat else 0.0,
        "routes": routes,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gateway", default="http://localhost:8010")
    p.add_argument("--label", required=True, help="e.g. metal-2slots, cpu-2slots")
    p.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--testset", type=Path, default=Path("eval/testset.jsonl"))
    p.add_argument("--target-p95", type=float, default=10.0, help="latency target for the summary")
    a = p.parse_args(argv)

    # Realistic mix: the main test set (lookups, planning, orders, out-of-scope, ambiguous).
    queries = [json.loads(line) for line in a.testset.read_text().splitlines()]
    url = a.gateway.rstrip("/") + "/v1/chat"
    out = {"label": a.label, "seconds_per_level": a.seconds, "levels": []}
    for u in a.users:
        res = asyncio.run(level(url, queries, u, a.seconds, a.timeout))
        out["levels"].append(res)
        print(json.dumps(res), flush=True)
    ok = [lv["users"] for lv in out["levels"] if lv["p95_s"] <= a.target_p95 and lv["errors"] == 0]
    out["max_users_p95_under_target"] = max(ok) if ok else 0
    out["target_p95_s"] = a.target_p95
    path = Path("eval/results") / f"loadtest_{a.label}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {path}; max users with p95 <= {a.target_p95}s: {out['max_users_p95_under_target']}")


if __name__ == "__main__":
    main()
