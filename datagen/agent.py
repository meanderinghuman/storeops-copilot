"""Run a model as a tool-calling agent against the real store tools and record the conversation.

The model only proposes tool calls; every call is executed for real by `StoreTools`, and the
actual result (or error) is what the model sees next. This is what makes the generated training
data execution-verified.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from store.prompt import system_prompt
from store.tools import TRAINING_TOOLS, StoreTools, ToolError, result_json, tool_specs

ChatFn = Callable[[list[dict], list[dict]], dict]  # (messages, tool specs) -> assistant message


def _assistant(msg: dict) -> dict:
    """Keep only what goes back to the model (drop reasoning, refusal, etc.)."""
    out = {"role": "assistant", "content": (msg.get("content") or "").strip()}
    calls = msg.get("tool_calls") or []
    if calls:
        out["tool_calls"] = [
            {
                "id": c.get("id") or f"call_{i}",
                "type": "function",
                "function": {
                    "name": c["function"]["name"],
                    "arguments": c["function"].get("arguments") or "{}",
                },
            }
            for i, c in enumerate(calls)
        ]
    return out


def run_agent(
    chat: ChatFn,
    tools: StoreTools,
    store_id: str,
    user_message: str,
    tool_names: list[str] = TRAINING_TOOLS,
    max_rounds: int = 4,
    history: list[dict] | None = None,
) -> dict:
    """One manager message -> full conversation with executed tool calls.

    `history` is earlier turns (after the system prompt) for follow-up requests; only calls made
    in response to the new message are recorded in "calls".
    Returns {"messages", "calls", "finish", "rounds", "seconds"}; finish is "answer" when the
    model ended with a plain reply, "max_rounds" when it was still calling tools at the cap.
    """
    specs = tool_specs(tool_names)
    messages = [
        {"role": "system", "content": system_prompt(store_id, tools.as_of)},
        *(history or []),
        {"role": "user", "content": user_message},
    ]
    calls, rounds, start = [], 0, time.monotonic()
    while True:
        msg = _assistant(chat(messages, specs))
        messages.append(msg)
        if "tool_calls" not in msg:
            finish = "answer"
            break
        if rounds == max_rounds:
            finish = "max_rounds"
            break
        rounds += 1
        for tc in msg["tool_calls"]:
            name, raw_args = tc["function"]["name"], tc["function"]["arguments"]
            record = {"round": rounds, "name": name, "arguments": raw_args}
            try:
                if name not in tool_names:
                    raise ToolError(f"Unknown tool '{name}'.")
                result = tools.execute(name, raw_args)
                record.update(ok=True, result=result)
            except ToolError as exc:
                result = {"error": str(exc)}
                record.update(ok=False, error=str(exc))
            calls.append(record)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result_json(result),
                }
            )
    return {
        "messages": messages,
        "calls": calls,
        "finish": finish,
        "rounds": rounds,
        "seconds": round(time.monotonic() - start, 2),
    }


def reset_writes(db_path: str | Path) -> None:
    """Clear staged orders/transfers so each conversation starts from the same state."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM purchase_orders")
        conn.execute("DELETE FROM stock_transfers")
        conn.commit()
    finally:
        conn.close()
