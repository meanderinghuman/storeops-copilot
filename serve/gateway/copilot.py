"""The copilot behind the gateway: agent loop + safety guards + escalation.

The model only proposes tool calls. This class decides what runs:

- Guards refuse a call before it runs: an item_id that no search_items call returned in this
  conversation (invented IDs), and purchase orders above a safety cap. The refusal goes back to the
  model like a tool error, so it can correct itself (e.g. search first).
- Writes only ever create *pending* orders/transfers; confirming is a separate API call the model
  cannot make.
- A malformed tool call gets one retry with a correction message.
- If the local attempt shows a problem (unrecovered tool error, refused call, round/time limit,
  bad output, a number not grounded in tool results), anything it staged is cancelled and the
  request is re-run on the large model. Without escalation, the local answer is returned and
  flagged `needs_review` (its writes stay pending, so nothing happens without the manager).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable

from datagen.agent import ChatFn, run_agent
from datagen.filters import plain, ungrounded_numbers
from datagen.teacher import Teacher
from serve.gateway import metrics as M
from serve.gateway.config import Settings
from store.tools import HELDOUT_TOOLS, TRAINING_TOOLS, WRITE_TOOLS, StoreTools, ToolError

ALL_TOOLS = TRAINING_TOOLS + HELDOUT_TOOLS
FALLBACK = (
    "Sorry, I couldn't complete that reliably. Please rephrase, or include the exact product name."
)
MALFORMED_HINT = (
    "Your previous reply contained a tool call that wasn't valid JSON. Reply again: either a "
    'valid <tool_call>{"name": ..., "arguments": {...}}</tool_call> or a plain answer.'
)

ChatFactory = Callable[[dict], ChatFn]  # usage accumulator -> chat function


def openai_chat_factory(base_url: str, api_key: str, model: str, timeout: float) -> ChatFactory:
    """Chat functions for an OpenAI-compatible endpoint, counting tokens per request."""

    def factory(usage: dict) -> ChatFn:
        def on_usage(u):
            usage["prompt"] = usage.get("prompt", 0) + u.get("prompt_tokens", 0)
            usage["completion"] = usage.get("completion", 0) + u.get("completion_tokens", 0)

        client = Teacher(base_url=base_url, api_key=api_key or "none", model=model,
                         timeout=timeout, on_usage=on_usage)  # fmt: skip

        def chat(messages, specs):
            return client.chat(messages, tools=specs, temperature=0.0, max_tokens=1024, retries=1)

        return chat

    return factory


class Copilot:
    def __init__(
        self,
        settings: Settings,
        tools: StoreTools,
        local: ChatFactory | None = None,
        escalation: ChatFactory | None = None,
    ):
        self.s = settings
        self.tools = tools
        self.tool_names = list(settings.tool_names) or ALL_TOOLS
        self.local = local or openai_chat_factory(
            settings.llm_base_url, "none", settings.llm_model, settings.request_timeout_s
        )
        if escalation is None and settings.escalation_enabled and settings.escalation_base_url:
            escalation = openai_chat_factory(
                settings.escalation_base_url, settings.escalation_api_key,
                settings.escalation_model, settings.request_timeout_s,
            )  # fmt: skip
        self.escalation = escalation if settings.escalation_enabled else None

    # -- guards ---------------------------------------------------------------------------------

    @staticmethod
    def searched_ids(messages: list[dict]) -> set[str]:
        """Item IDs returned by search_items anywhere in the conversation (incl. earlier turns)."""
        ids = set()
        for m in messages:
            if m["role"] == "tool" and m.get("name") == "search_items":
                try:
                    ids.update(x["item_id"] for x in json.loads(m["content"]).get("matches", []))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
        return ids

    def quantity_cap(self, item_id: str, store_id: str) -> int:
        inv = self.tools.get_inventory(item_id, store_id)
        p90 = self.tools.get_forecast(item_id, store_id, inv["lead_time_days"] + 7)["total"]["p90"]
        return int(
            max(self.s.qty_cap_p90_multiple * p90, self.s.qty_cap_min_packs * inv["pack_size"])
        )

    def guard(self, name: str, raw_args, messages: list[dict]) -> str | None:
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError):
            return None  # let schema validation report it
        if not isinstance(args, dict):
            return None
        item = args.get("item_id")
        if isinstance(item, str) and item not in self.searched_ids(messages):
            M.REJECTED.labels("unsearched_id").inc()
            return (
                f"item_id '{item}' did not come from search_items in this conversation. Call "
                "search_items with the product name first, then use an item_id from its results."
            )
        if name == "create_purchase_order" and isinstance(args.get("quantity"), int):
            try:
                cap = self.quantity_cap(item, args.get("store_id", ""))
            except ToolError:
                return None  # unknown store/item: the tool itself reports it
            if args["quantity"] > cap:
                M.REJECTED.labels("quantity_cap").inc()
                return (
                    f"Quantity {args['quantity']} is above the safety cap of {cap} units for this "
                    "item (5x high-case demand over lead time + review, at least 4 packs). Tell "
                    "the manager it needs purchasing approval, or suggest a smaller quantity."
                )
        return None

    # -- one attempt ----------------------------------------------------------------------------

    def _with_retry(self, chat: ChatFn) -> ChatFn:
        def wrapped(messages, specs):
            msg = chat(messages, specs)
            if not msg.get("tool_calls") and "<tool_call>" in (msg.get("content") or ""):
                M.RETRIES.inc()
                msg = chat([*messages, {"role": "user", "content": MALFORMED_HINT}], specs)
            return msg

        return wrapped

    def _attempt(self, factory: ChatFactory, store_id, message, history) -> tuple[dict, dict]:
        usage: dict = {}
        conv = run_agent(
            self._with_retry(factory(usage)), self.tools, store_id, message, self.tool_names,
            self.s.max_rounds, history, self.guard, time.monotonic() + self.s.request_timeout_s,
        )  # fmt: skip
        return conv, usage

    @staticmethod
    def problems(conv: dict) -> list[str]:
        """Reasons the attempt can't be trusted as is (empty = fine)."""
        out = []
        if conv["finish"] != "answer":
            out.append(conv["finish"])  # max_rounds / timeout
        answer = (conv["messages"][-1].get("content") or "") if conv["finish"] == "answer" else ""
        if conv["finish"] == "answer" and (not answer.strip() or "<tool_call>" in answer):
            out.append("bad_output")
        calls = conv["calls"]
        last_error = max((i for i, c in enumerate(calls) if not c["ok"]), default=None)
        if last_error is not None and not any(c["ok"] for c in calls[last_error + 1 :]):
            out.append("refused_call" if calls[last_error].get("guard") else "tool_error")
        if answer and ungrounded_numbers(plain(answer), conv["messages"]):
            out.append("ungrounded_number")
        return out

    def _cancel_writes(self, conv: dict) -> None:
        for c in conv["calls"]:
            if c["name"] in WRITE_TOOLS and c.get("ok"):
                res = c["result"]
                if "po_id" in res:
                    self.tools.cancel_purchase_order(res["po_id"])
                else:
                    self.tools.cancel_stock_transfer(res["transfer_id"])
                M.ORDERS.labels(c["name"], "auto_cancelled").inc()

    # -- request --------------------------------------------------------------------------------

    def handle(self, store_id: str, message: str, history: list[dict] | None = None) -> dict:
        """Answer one manager message. Returns the answer, route, trace and pending writes."""
        start, request_id = time.monotonic(), uuid.uuid4().hex[:12]
        route, reasons, usage = "local", [], {}
        try:
            conv, usage = self._attempt(self.local, store_id, message, history)
            reasons = self.problems(conv)
        except Exception as exc:  # noqa: BLE001 - local model down or broken: escalate
            conv, reasons = None, [f"local_error:{type(exc).__name__}"]
        local_conv = conv

        if reasons and self.escalation:
            for r in reasons:
                M.ESCALATIONS.labels(r.split(":")[0]).inc()
            if conv:
                self._cancel_writes(conv)
            try:
                conv, usage_e = self._attempt(self.escalation, store_id, message, history)
                route = "escalated"
                for k, v in usage_e.items():
                    M.TOKENS.labels("escalated", k).inc(v)
            except Exception:  # noqa: BLE001 - large model unavailable: safe fallback below
                conv, route = None, "fallback"
        if conv is None:
            route = "fallback"
        for k, v in usage.items():
            M.TOKENS.labels("local", k).inc(v)

        answer = FALLBACK if conv is None else (conv["messages"][-1].get("content") or FALLBACK)
        calls = conv["calls"] if conv else []
        for c in calls:
            M.TOOL_CALLS.labels(c["name"], "ok" if c["ok"] else "error").inc()
            if not c["ok"] and not c.get("guard"):
                M.REJECTED.labels("tool_error").inc()
        writes = [c["result"] for c in calls if c["name"] in WRITE_TOOLS and c.get("ok")]
        for c in calls:
            if c["name"] in WRITE_TOOLS and c.get("ok"):
                M.ORDERS.labels(c["name"], "staged").inc()
        seconds = round(time.monotonic() - start, 3)
        M.LATENCY.labels(route).observe(seconds)
        M.REQUESTS.labels(route, "ok" if route != "fallback" else "fallback").inc()
        return {
            "request_id": request_id,
            "answer": answer,
            "route": route,
            "escalation_reasons": reasons,  # why the local attempt wasn't trusted (empty = fine)
            "needs_review": bool(reasons) and route == "local",
            "trace": [
                {k: c.get(k) for k in ("name", "arguments", "ok", "result", "error")} for c in calls
            ],
            "pending": writes,
            "seconds": seconds,
            "conversation": conv,
            "local_conversation": local_conv if route == "escalated" else None,
        }
