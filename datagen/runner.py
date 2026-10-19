"""Run many agent conversations in parallel, each worker on its own scratch copy of the store DB."""

from __future__ import annotations

import json
import shutil
import threading
import traceback
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datagen.agent import reset_writes, run_agent
from datagen.filters import check
from datagen.teacher import Teacher
from store.tools import StoreTools


def run_many(
    tasks: Iterable[dict],
    model: str,
    db_path: Path,
    scratch_dir: Path,
    workers: int = 4,
    temperature: float = 0.3,
    max_tokens: int = 4000,
) -> Iterator[dict]:
    """Yield one record per task ({"id", "store_id", "kind", "query"}) as conversations finish.

    Each record gets "model", "conversation" (or "api_error") and "reasons" (empty = kept).
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(model=model, timeout=60.0)  # agent turns normally take 1-15 s
    local = threading.local()

    def worker_tools() -> StoreTools:
        if not hasattr(local, "tools"):
            path = scratch_dir / f"worker_{threading.get_ident()}.db"
            shutil.copyfile(db_path, path)
            local.tools = StoreTools(path)
        return local.tools

    def chat(messages: list[dict], specs: list[dict]) -> dict:
        return teacher.chat(messages, tools=specs, temperature=temperature, max_tokens=max_tokens)

    def one(task: dict) -> dict:
        tools = worker_tools()
        reset_writes(tools.db_path)
        rec = {**task, "model": model}
        try:
            conv = run_agent(chat, tools, task["store_id"], task["query"])
        except Exception as exc:  # noqa: BLE001 - record and move on; one bad call shouldn't stop a run
            rec.update(api_error=f"{type(exc).__name__}: {exc}"[:500], reasons=["api_error"])
            rec["traceback"] = traceback.format_exc(limit=3)
            return rec
        rec.update(conversation=conv, reasons=check(conv, task["kind"]))
        return rec

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(one, t) for t in tasks]
        for fut in as_completed(futures):
            yield fut.result()
    finally:  # caller stopped early (e.g. API unreachable): drop queued work
        pool.shutdown(wait=True, cancel_futures=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
