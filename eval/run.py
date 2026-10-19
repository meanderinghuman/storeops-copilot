"""Score a model on the test set through the same agent loop used everywhere else.

    # the large API model (teacher endpoint from .env)
    uv run python -m eval.run --model DeepSeek-V4.1-Flash
    # a local GGUF behind llama-server (no key needed)
    uv run python -m eval.run --model qwen2.5-1.5b-base --base-url http://localhost:8080/v1 --api-key none

Every scenario runs at temperature 0 with all 8 tools (6 training + 2 held out), on a scratch copy
of the store DB that is reset before each scenario. Transcripts go to data/eval/<label>.jsonl
(resumable); the metrics summary goes to eval/results/<label>.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datagen.agent import reset_writes, run_agent
from datagen.runner import append_jsonl, read_jsonl
from datagen.teacher import Teacher
from eval.metrics import aggregate, score
from store.tools import HELDOUT_TOOLS, TRAINING_TOOLS, StoreTools

ALL_TOOLS = TRAINING_TOOLS + HELDOUT_TOOLS


def run(scenarios, model, base_url, api_key, db, scratch, workers, max_rounds=4):
    scratch.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def worker_tools() -> StoreTools:
        if not hasattr(local, "tools"):
            path = scratch / f"eval_{threading.get_ident()}.db"
            shutil.copyfile(db, path)
            local.tools = StoreTools(path)
        return local.tools

    def one(sc: dict) -> dict:
        tools = worker_tools()
        reset_writes(tools.db_path)
        tokens = {"prompt": 0, "completion": 0}

        def on_usage(u):
            tokens["prompt"] += u.get("prompt_tokens", 0)
            tokens["completion"] += u.get("completion_tokens", 0)

        client = Teacher(
            base_url=base_url, api_key=api_key, model=model, timeout=120.0, on_usage=on_usage
        )

        def chat(messages, specs):
            return client.chat(messages, tools=specs, temperature=0.0, max_tokens=1024)

        try:
            conv = run_agent(chat, tools, sc["store_id"], sc["query"], ALL_TOOLS, max_rounds)
        except Exception as exc:  # noqa: BLE001 - recorded, retried on the next run
            return {"id": sc["id"], "api_error": f"{type(exc).__name__}: {exc}"[:300]}
        conv["tokens"] = tokens["prompt"] + tokens["completion"] or None
        conv["prompt_tokens"], conv["completion_tokens"] = tokens["prompt"], tokens["completion"]
        return {"id": sc["id"], "conversation": conv, "score": score(sc, conv)}

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(one, sc) for sc in scenarios]
        for fut in as_completed(futures):
            yield fut.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", help="results name (default: model)")
    p.add_argument("--base-url", default=None, help="default TEACHER_BASE_URL")
    p.add_argument(
        "--api-key", default=None, help="default TEACHER_API_KEY; 'none' for llama-server"
    )
    p.add_argument("--testset", type=Path, default=Path("eval/testset.jsonl"))
    p.add_argument("--db", type=Path, default=Path("data/store.db"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)

    label = a.label or a.model
    out = Path("data/eval") / f"{label}.jsonl"
    scenarios = {sc["id"]: sc for sc in read_jsonl(a.testset)[: a.limit]}
    done = {r["id"]: r for r in read_jsonl(out) if "score" in r}
    todo = [sc for i, sc in scenarios.items() if i not in done]
    print(f"{label}: {len(scenarios)} scenarios, {len(done)} done, {len(todo)} to run")

    start, errors_in_a_row = time.monotonic(), 0
    for n, rec in enumerate(
        run(todo, a.model, a.base_url, a.api_key, a.db, Path("data/eval/scratch"), a.workers), 1
    ):
        append_jsonl(out, rec)
        errors_in_a_row = errors_in_a_row + 1 if "api_error" in rec else 0
        if errors_in_a_row >= 10:
            print(f"stopping: 10 API errors in a row. Last: {rec['api_error']}")
            break
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {n / (time.monotonic() - start) * 60:.0f}/min", flush=True)

    results = {r["id"]: r for r in read_jsonl(out) if "score" in r}
    scores = [results[i]["score"] for i in scenarios if i in results]
    summary = {"label": label, "model": a.model, **aggregate(scores)}
    missing = len(scenarios) - len(scores)
    if missing:
        summary["incomplete"] = f"{missing} scenarios not scored (API errors); re-run to resume"
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    (Path("eval/results") / f"{label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
