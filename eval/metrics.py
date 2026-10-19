"""Score a model's conversation against a test scenario, and aggregate over a run.

Step level (each tool call): valid call rate, tool selection, exact argument match, hallucinated IDs.
Task level (whole conversation): task success, faithfulness, decline accuracy, clarification rate,
unrequested writes, steps per task. Plus latency and tokens when the run recorded them.
"""

from __future__ import annotations

import json
import random
import re
import statistics

from datagen.filters import numbers_in, plain, ungrounded_numbers
from store.tools import WRITE_TOOLS, apply_defaults

INVALID = re.compile(
    r"^(Unknown tool|Invalid arguments|Arguments are not valid JSON|Arguments must be)"
)
HALLUCINATED = re.compile(r"^Unknown (item_id|store_id)")


def _args(call: dict) -> dict:
    raw = call["arguments"]
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


def _norm(name: str, args: dict) -> dict:
    try:
        return apply_defaults(name, args)
    except KeyError:  # tool the model invented
        return args


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) < 1e-9
    return a == b


def _matches(expected: dict, call: dict) -> tuple[bool, bool]:
    """(right tool, right tool with all expected arguments)."""
    names = expected["name"] if isinstance(expected["name"], list) else [expected["name"]]
    if call["name"] not in names:
        return False, False
    got = _norm(call["name"], _args(call))
    return True, all(k in got and _same(got[k], v) for k, v in expected["args"].items())


def number_present(value: float, answer: str) -> bool:
    """The expected number appears in the answer, exactly or rounded to an integer."""
    return any(
        _same(v, value) or (v.is_integer() and abs(value - v) <= 0.5) for v, _ in numbers_in(answer)
    )


def score(scenario: dict, conv: dict) -> dict:
    exp = scenario["expected"]
    calls = conv["calls"]
    answer = plain(conv["messages"][-1].get("content") or "") if conv["finish"] == "answer" else ""
    writes = [c for c in calls if c["name"] in WRITE_TOOLS and c.get("ok")]

    invalid = sum(1 for c in calls if not c.get("ok") and INVALID.match(c.get("error", "")))
    hallucinated = sum(
        1 for c in calls if not c.get("ok") and HALLUCINATED.match(c.get("error", ""))
    )

    wanted = exp["calls"] + ([exp["write"]] if exp["write"] else [])
    selected = exact = 0
    for w in wanted:
        hits = [_matches(w, c) for c in calls if c.get("ok") or w is exp["write"]]
        selected += any(h[0] for h in hits)
        exact += any(h[1] for h in hits)

    checks = {
        "finished": conv["finish"] == "answer" and bool(answer.strip()),
        "calls": exact == len(wanted),
        "numbers": all(number_present(float(n), answer) for n in exp["answer_numbers"]),
        "terms": not exp["answer_terms_any"]
        or any(t.lower() in answer.lower() for t in exp["answer_terms_any"]),
    }
    outcome = exp["outcome"]
    if outcome in ("order", "transfer"):
        checks["one_write"] = len(writes) == 1
        checks["asks_confirmation"] = bool(re.search(r"confirm", answer, re.IGNORECASE))
    else:
        checks["no_write"] = not writes
    if outcome == "clarify":
        checks["asks_question"] = "?" in answer
    if outcome == "decline":
        checks["no_tool_calls"] = not calls

    return {
        "id": scenario["id"],
        "split": scenario["split"],
        "category": scenario["category"],
        "task_success": all(checks.values()),
        "failed_checks": [k for k, ok in checks.items() if not ok],
        "n_calls": len(calls),
        "invalid_calls": invalid,
        "hallucinated_id_calls": hallucinated,
        "expected_calls": len(wanted),
        "tool_selected": selected,
        "exact_args": exact,
        "unrequested_write": bool(writes) and exp["write"] is None,
        "faithful": not answer or not ungrounded_numbers(answer, conv["messages"]),
        "seconds": conv.get("seconds"),
        "tokens": conv.get("tokens"),
    }


def bootstrap_ci(values: list[bool], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for a success rate."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return (round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3))


def aggregate(results: list[dict]) -> dict:
    def rate(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 3) if xs else None

    calls = sum(r["n_calls"] for r in results)
    wanted = sum(r["expected_calls"] for r in results)
    out = {
        "scenarios": len(results),
        "task_success": rate(r["task_success"] for r in results),
        "task_success_ci95": bootstrap_ci([r["task_success"] for r in results]),
        "by_split": {},
        "valid_call_rate": round(1 - sum(r["invalid_calls"] for r in results) / calls, 3) if calls else None,
        "hallucinated_id_rate": round(sum(r["hallucinated_id_calls"] for r in results) / calls, 3) if calls else None,
        "tool_selection_accuracy": round(sum(r["tool_selected"] for r in results) / wanted, 3) if wanted else None,
        "exact_arg_match": round(sum(r["exact_args"] for r in results) / wanted, 3) if wanted else None,
        "faithfulness": rate(r["faithful"] for r in results if r["split"] != "out_of_scope"),
        "decline_accuracy": rate(r["task_success"] for r in results if r["split"] == "out_of_scope"),
        "clarification_rate": rate(
            "asks_question" not in r["failed_checks"] and "no_write" not in r["failed_checks"]
            for r in results
            if r["category"].startswith("ambiguous")
        ),
        "unrequested_write_rate": rate(r["unrequested_write"] for r in results),
        "steps_per_task": round(statistics.mean(r["n_calls"] for r in results), 2) if results else None,
    }  # fmt: skip
    for split in ("seen", "unseen", "out_of_scope", "ambiguous_safety"):
        rs = [r["task_success"] for r in results if r["split"] == split]
        if rs:
            out["by_split"][split] = {
                "n": len(rs),
                "task_success": rate(rs),
                "ci95": bootstrap_ci(rs),
            }
    secs = sorted(r["seconds"] for r in results if r["seconds"] is not None)
    if secs:
        out["latency_p50_s"] = round(statistics.median(secs), 2)
        out["latency_p95_s"] = round(secs[min(len(secs) - 1, int(0.95 * len(secs)))], 2)
    toks = [r["tokens"] for r in results if r["tokens"]]
    if toks:
        out["tokens_per_conversation"] = round(statistics.mean(toks))
    return out
