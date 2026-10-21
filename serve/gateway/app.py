"""StoreOps Copilot gateway (FastAPI).

    uv run uvicorn serve.gateway.app:app --port 8000

POST /v1/chat                         manager message -> answer, tool trace, pending orders
GET  /v1/orders?store_id=&status=     purchase orders (default: pending)
POST /v1/orders/{po_id}/confirm       manager confirms (the model can never call this)
POST /v1/orders/{po_id}/cancel
POST /v1/transfers/{id}/confirm|cancel
GET  /health  /ready  /metrics

If GATEWAY_API_KEY is set, every /v1 route needs the header X-API-Key.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from serve.gateway import metrics as M
from serve.gateway.config import Settings
from serve.gateway.copilot import Copilot
from store.tools import StoreTools, ToolError

log = logging.getLogger("copilot")
WEB = Path(__file__).resolve().parents[1] / "web"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        data.update(getattr(record, "fields", {}))
        return json.dumps(data)


class ChatRequest(BaseModel):
    store_id: str = Field(pattern=r"^[A-Z]{2}_\d+$")
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=40)


class RateLimiter:
    """Sliding one-minute window per client key."""

    def __init__(self, per_minute: int):
        self.per_minute, self.hits, self.lock = per_minute, defaultdict(deque), threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def create_app(settings: Settings | None = None, copilot: Copilot | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    tools = copilot.tools if copilot else StoreTools(s.db_path, s.forecast_url)
    bot = copilot or Copilot(s, tools)
    limiter = RateLimiter(s.rate_limit_per_min)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)

    app = FastAPI(title="StoreOps Copilot gateway", version="1.0")

    def auth(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        if s.api_key and x_api_key != s.api_key:
            raise HTTPException(401, "missing or invalid X-API-Key")
        client = x_api_key or (request.client.host if request.client else "anon")
        if not limiter.allow(client):
            raise HTTPException(429, "rate limit exceeded")
        return client

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        checks = {}
        try:
            with sqlite3.connect(f"file:{tools.db_path}?mode=ro", uri=True) as conn:
                conn.execute("SELECT 1 FROM items LIMIT 1")
            checks["store_db"] = "ok"
        except sqlite3.Error as exc:
            checks["store_db"] = f"error: {exc}"
        try:
            r = httpx.get(s.llm_base_url.rstrip("/").removesuffix("/v1") + "/health", timeout=3)
            checks["local_llm"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except httpx.HTTPError as exc:
            checks["local_llm"] = f"error: {type(exc).__name__}"
        checks["escalation"] = "configured" if bot.escalation else "off"
        ok = checks["store_db"] == "ok" and (checks["local_llm"] == "ok" or bot.escalation)
        if not ok:
            raise HTTPException(503, checks)
        return checks

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat")
    def chat(req: ChatRequest, client: str = Depends(auth)) -> dict:
        out = bot.handle(req.store_id, req.message, req.history or None)
        log.info("chat", extra={"fields": {
            "request_id": out["request_id"], "store_id": req.store_id, "route": out["route"],
            "reasons": out["escalation_reasons"], "seconds": out["seconds"],
            "tools": [t["name"] for t in out["trace"]], "pending": len(out["pending"]),
        }})  # fmt: skip
        resp = {k: v for k, v in out.items() if k not in ("conversation", "local_conversation")}
        # This turn as structured messages (incl. tool calls and results), so a client can send it
        # back as `history` for a follow-up like "yes, go ahead" and the model sees the item IDs.
        msgs = out["conversation"]["messages"] if out["conversation"] else []
        start = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=None)
        resp["turn"] = (
            msgs[start:]
            if start is not None
            else [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": out["answer"]},
            ]
        )
        return resp

    def listing(table: str, store_col: str, store_id: str | None, status: str) -> list[dict]:
        q = (f"SELECT t.*, i.name FROM {table} t JOIN items i USING (item_id) WHERE t.status = ?"
             + (f" AND t.{store_col} = ?" if store_id else "") + " ORDER BY t.created_at")  # fmt: skip
        with sqlite3.connect(f"file:{tools.db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(q, (status, store_id) if store_id else (status,))]

    @app.get("/v1/orders")
    def orders(
        store_id: str | None = None, status: str = "pending", _: str = Depends(auth)
    ) -> list:
        return listing("purchase_orders", "store_id", store_id, status)

    @app.get("/v1/transfers")
    def transfers(
        store_id: str | None = None, status: str = "pending", _: str = Depends(auth)
    ) -> list:
        """Transfers touching the store (sending or receiving)."""
        rows = listing("stock_transfers", "from_store", None, status)
        return [r for r in rows if not store_id or store_id in (r["from_store"], r["to_store"])]

    def manager_action(fn, kind: str, action: str, ref: str) -> dict:
        try:
            out = fn(ref)
        except ToolError as exc:
            raise HTTPException(409, str(exc)) from exc
        M.ORDERS.labels(kind, action).inc()
        log.info(f"{kind} {action}", extra={"fields": {"ref": ref}})
        return out

    @app.post("/v1/orders/{po_id}/confirm")
    def confirm_order(po_id: str, _: str = Depends(auth)) -> dict:
        return manager_action(
            tools.confirm_purchase_order, "create_purchase_order", "confirmed", po_id
        )

    @app.post("/v1/orders/{po_id}/cancel")
    def cancel_order(po_id: str, _: str = Depends(auth)) -> dict:
        return manager_action(
            tools.cancel_purchase_order, "create_purchase_order", "cancelled", po_id
        )

    @app.post("/v1/transfers/{transfer_id}/confirm")
    def confirm_transfer(transfer_id: str, _: str = Depends(auth)) -> dict:
        return manager_action(
            tools.confirm_stock_transfer, "request_stock_transfer", "confirmed", transfer_id
        )

    @app.post("/v1/transfers/{transfer_id}/cancel")
    def cancel_transfer(transfer_id: str, _: str = Depends(auth)) -> dict:
        return manager_action(
            tools.cancel_stock_transfer, "request_stock_transfer", "cancelled", transfer_id
        )

    # Web UI: a static single-page app served by the gateway itself (same origin, no build step).
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


try:
    app = create_app()
except FileNotFoundError:  # no store DB yet (e.g. CI); tests build their own app
    app = None
