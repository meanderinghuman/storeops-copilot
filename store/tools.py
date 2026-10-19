"""Store tools the copilot can call: JSON Schemas plus their implementations.

Six training tools and two held-out tools (test-time only). Read tools use a read-only SQLite
connection. Write tools only ever create *pending* records; confirming or cancelling is done by
`confirm_*` / `cancel_*`, which the gateway exposes to the manager's UI and never to the model.
All quantity math (reorder suggestion) is done here in code, not by the model.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import httpx
import jsonschema
from dotenv import load_dotenv

from store.forecast import MAX_HORIZON_DAYS, Z90, UnknownSeries, seasonal_naive

ITEM_ID = {
    "type": "string",
    "pattern": r"^[A-Z]+_\d+_\d+$",
    "description": "Item ID from search_items, e.g. HOUSEHOLD_1_118.",
}
STORE_ID = {"type": "string", "pattern": r"^[A-Z]{2}_\d+$", "description": "Store ID, e.g. CA_1."}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: dict[str, dict] = {
    s["function"]["name"]: s
    for s in [
        _fn(
            "search_items",
            "Find products by name at a store. Use this first to get the item_id for any product "
            "the manager mentions. Returns up to 5 matches with item IDs, names and prices.",
            {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Product words, e.g. 'paper towels'.",
                },
                "store_id": STORE_ID,
            },
            ["query", "store_id"],
        ),
        _fn(
            "get_inventory",
            "Current stock for one item at a store: units on hand, units on order (in transit), "
            "units in pending purchase orders, supplier lead time in days, and pack size.",
            {"item_id": ITEM_ID, "store_id": STORE_ID},
            ["item_id", "store_id"],
        ),
        _fn(
            "get_sales_history",
            "Units sold per week for one item at a store over the most recent weeks.",
            {
                "item_id": ITEM_ID,
                "store_id": STORE_ID,
                "weeks": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 52,
                    "default": 4,
                    "description": "Number of recent weeks (default 4).",
                },
            },
            ["item_id", "store_id"],
        ),
        _fn(
            "get_forecast",
            "Expected demand (units) for one item at a store over the next N days, as p10 (low), "
            "p50 (expected) and p90 (high) totals plus a daily breakdown.",
            {
                "item_id": ITEM_ID,
                "store_id": STORE_ID,
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_HORIZON_DAYS,
                    "default": 7,
                    "description": "Forecast horizon in days starting today (default 7).",
                },
            },
            ["item_id", "store_id"],
        ),
        _fn(
            "get_reorder_suggestion",
            "Suggested order quantity for one item at a store, computed from the forecast over "
            "lead time plus a 7-day review period, current stock and pack size. Returns the "
            "quantity and the numbers behind it. Does not place an order.",
            {
                "item_id": ITEM_ID,
                "store_id": STORE_ID,
                "service_level": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 0.99,
                    "default": 0.95,
                    "description": "Chance of not running out before the next order arrives "
                    "(default 0.95).",
                },
            },
            ["item_id", "store_id"],
        ),
        _fn(
            "create_purchase_order",
            "Stage a purchase order. The order is created as PENDING and is only placed after the "
            "manager confirms it in the app. Only use when the manager asked to order.",
            {
                "item_id": ITEM_ID,
                "store_id": STORE_ID,
                "quantity": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                    "description": "Units to order.",
                },
            },
            ["item_id", "store_id", "quantity"],
        ),
        _fn(
            "get_promotions",
            "Price promotions at a store starting or running within the next N days.",
            {
                "store_id": STORE_ID,
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_HORIZON_DAYS,
                    "default": 7,
                    "description": "Look-ahead window in days (default 7).",
                },
            },
            ["store_id"],
        ),
        _fn(
            "request_stock_transfer",
            "Stage a transfer of units of one item from one store to another. The transfer is "
            "created as PENDING and only happens after the manager confirms it in the app. Only "
            "use when the manager asked to move stock.",
            {
                "item_id": ITEM_ID,
                "from_store": STORE_ID,
                "to_store": STORE_ID,
                "quantity": {"type": "integer", "minimum": 1, "maximum": 10000},
            },
            ["item_id", "from_store", "to_store", "quantity"],
        ),
    ]
}

TRAINING_TOOLS = [
    "search_items",
    "get_inventory",
    "get_sales_history",
    "get_forecast",
    "get_reorder_suggestion",
    "create_purchase_order",
]
HELDOUT_TOOLS = ["get_promotions", "request_stock_transfer"]
WRITE_TOOLS = {"create_purchase_order", "request_stock_transfer"}
REVIEW_PERIOD_DAYS = 7


def result_json(result: dict) -> str:
    """How a tool result is shown to the model: compact JSON (saves ~15% of result tokens)."""
    return json.dumps(result, separators=(",", ":"))


def tool_specs(names: list[str] = TRAINING_TOOLS) -> list[dict]:
    """OpenAI / Qwen-style tool list for the system prompt."""
    return [TOOL_SCHEMAS[n] for n in names]


def apply_defaults(name: str, args: dict) -> dict:
    """Fill schema defaults, so `{'item_id': x, 'store_id': y}` and the explicit form compare equal."""
    props = TOOL_SCHEMAS[name]["function"]["parameters"]["properties"]
    return {**{k: p["default"] for k, p in props.items() if "default" in p}, **args}


class ToolError(Exception):
    """A tool call that can't be executed; the message is safe to show to the model."""


ForecastFn = Callable[[str, str, int], dict]


class StoreTools:
    def __init__(self, db_path: str | Path, forecast_url: str | None = None):
        """`forecast_url` set -> call the forecast service over HTTP; None -> compute in-process
        with the same function the stub uses (identical numbers, no server needed)."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"{self.db_path} not found; run `python -m store.build_db`")
        self.forecast_url = forecast_url.rstrip("/") if forecast_url else None
        with self._read() as conn:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
        self.as_of = date.fromisoformat(meta["as_of_date"])

    @classmethod
    def from_env(cls) -> StoreTools:
        load_dotenv()
        return cls(os.getenv("STORE_DB_PATH", "data/store.db"), os.getenv("FORECAST_URL") or None)

    # -- connections ------------------------------------------------------------------------

    @contextmanager
    def _read(self):
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write(self):
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # -- dispatch ---------------------------------------------------------------------------

    def execute(self, name: str, arguments: dict | str) -> dict:
        """Validate a model's tool call against its schema and run it. Raises ToolError."""
        if name not in TOOL_SCHEMAS:
            raise ToolError(f"Unknown tool '{name}'.")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolError(f"Arguments are not valid JSON: {exc.msg}.") from exc
        if not isinstance(arguments, dict):
            raise ToolError("Arguments must be a JSON object.")
        schema = TOOL_SCHEMAS[name]["function"]["parameters"]
        try:
            jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as exc:
            where = ".".join(str(p) for p in exc.absolute_path) or "arguments"
            raise ToolError(f"Invalid arguments for {name} ({where}): {exc.message}") from exc
        return getattr(self, name)(**apply_defaults(name, arguments))

    # -- lookups ----------------------------------------------------------------------------

    def _check_store(self, conn, store_id: str) -> None:
        if not conn.execute("SELECT 1 FROM stores WHERE store_id = ?", (store_id,)).fetchone():
            raise ToolError(f"Unknown store_id '{store_id}'.")

    def _item(self, conn, item_id: str, store_id: str) -> sqlite3.Row:
        self._check_store(conn, store_id)
        row = conn.execute(
            "SELECT i.item_id, i.name, i.pack_size, v.on_hand, v.on_order, v.lead_time_days "
            "FROM items i JOIN inventory v ON v.item_id = i.item_id AND v.store_id = ? "
            "WHERE i.item_id = ?",
            (store_id, item_id),
        ).fetchone()
        if row is None:
            raise ToolError(f"Unknown item_id '{item_id}' at store {store_id}.")
        return row

    def _forecast(self, item_id: str, store_id: str, days: int) -> dict:
        if self.forecast_url is None:
            with self._read() as conn:
                try:
                    return seasonal_naive(conn, item_id, store_id, days)
                except UnknownSeries as exc:
                    raise ToolError(str(exc)) from exc
        try:
            resp = httpx.post(
                f"{self.forecast_url}/forecast",
                json={"item_id": item_id, "store_id": store_id, "horizon_days": days},
                timeout=10,
            )
        except httpx.HTTPError as exc:
            raise ToolError("Forecast service is unavailable.") from exc
        if resp.status_code != 200:
            raise ToolError(f"Forecast service error {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # -- training tools ---------------------------------------------------------------------

    def search_items(self, query: str, store_id: str) -> dict:
        def tokens(text: str) -> list[str]:
            words = re.findall(r"[a-z0-9]+(?:/[0-9]+)?", text.lower())
            return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]

        q = [t for t in tokens(query) if len(t) > 1]
        with self._read() as conn:
            self._check_store(conn, store_id)
            rows = conn.execute(
                "SELECT i.item_id, i.name, i.dept_id, p.price FROM items i "
                "JOIN inventory v ON v.item_id = i.item_id AND v.store_id = ? "
                "LEFT JOIN prices p ON p.item_id = i.item_id AND p.store_id = v.store_id",
                (store_id,),
            ).fetchall()
        exact = [r for r in rows if r["item_id"].lower() == query.strip().lower()]
        scored = []
        for r in exact or rows:
            name_toks = tokens(r["name"])
            hits = 0.0
            for t in q:
                if t in name_toks:
                    hits += 1
                elif any(n.startswith(t) or (len(n) >= 3 and t.startswith(n)) for n in name_toks):
                    hits += 0.8
                elif len(t) >= 4 and difflib.get_close_matches(t, name_toks, n=1, cutoff=0.8):
                    hits += 0.7  # typo, e.g. "towles"
            score = 1.0 if exact else hits / max(len(q), 1)
            if score >= 0.5:
                scored.append((-score, r["name"], r))
        scored.sort(key=lambda s: (s[0], s[1]))
        matches = [
            {
                "item_id": r["item_id"],
                "name": r["name"],
                "department": r["dept_id"],
                "price": r["price"],
            }
            for _, _, r in scored[:5]
        ]
        return {"store_id": store_id, "query": query, "matches": matches}

    def get_inventory(self, item_id: str, store_id: str) -> dict:
        with self._read() as conn:
            row = self._item(conn, item_id, store_id)
            pending = conn.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM purchase_orders "
                "WHERE item_id = ? AND store_id = ? AND status = 'pending'",
                (item_id, store_id),
            ).fetchone()[0]
        return {
            "item_id": item_id,
            "name": row["name"],
            "store_id": store_id,
            "on_hand": row["on_hand"],
            "on_order": row["on_order"],
            "pending_po_quantity": pending,
            "lead_time_days": row["lead_time_days"],
            "pack_size": row["pack_size"],
            "as_of_date": self.as_of.isoformat(),
        }

    def get_sales_history(self, item_id: str, store_id: str, weeks: int) -> dict:
        start = self.as_of - timedelta(days=7 * weeks - 1)
        with self._read() as conn:
            row = self._item(conn, item_id, store_id)
            daily = dict(
                conn.execute(
                    "SELECT date, units FROM sales WHERE item_id = ? AND store_id = ? AND date >= ?",
                    (item_id, store_id, start.isoformat()),
                ).fetchall()
            )
        out = []
        for w in range(weeks):
            ws = start + timedelta(days=7 * w)
            days = [(ws + timedelta(days=d)).isoformat() for d in range(7)]
            out.append(
                {
                    "week_start": days[0],
                    "week_end": days[-1],
                    "units": sum(daily.get(d, 0) for d in days),
                }
            )
        total = sum(w["units"] for w in out)
        return {
            "item_id": item_id,
            "name": row["name"],
            "store_id": store_id,
            "weeks": out,
            "total_units": total,
            "avg_units_per_week": round(total / weeks, 1),
        }

    def get_forecast(self, item_id: str, store_id: str, days: int) -> dict:
        with self._read() as conn:
            row = self._item(conn, item_id, store_id)
        fc = self._forecast(item_id, store_id, days)
        return {"name": row["name"], **fc}

    def get_reorder_suggestion(self, item_id: str, store_id: str, service_level: float) -> dict:
        inv = self.get_inventory(item_id, store_id)
        coverage = inv["lead_time_days"] + REVIEW_PERIOD_DAYS
        fc = self._forecast(item_id, store_id, coverage)["total"]
        # Recover the normal approximation from the quantiles (upper side: p10 may be floored at 0).
        mean, sd = fc["p50"], (fc["p90"] - fc["p50"]) / Z90
        demand = round(max(0.0, mean + NormalDist().inv_cdf(service_level) * sd), 1)
        available = inv["on_hand"] + inv["on_order"] + inv["pending_po_quantity"]
        net_need = round(demand - available, 1)
        pack = inv["pack_size"]
        packs = math.ceil(net_need / pack) if net_need > 0 else 0
        return {
            "item_id": item_id,
            "name": inv["name"],
            "store_id": store_id,
            "service_level": service_level,
            "lead_time_days": inv["lead_time_days"],
            "review_period_days": REVIEW_PERIOD_DAYS,
            "coverage_days": coverage,
            "forecast_p50": fc["p50"],
            "demand_at_service_level": demand,
            "on_hand": inv["on_hand"],
            "on_order": inv["on_order"],
            "pending_po_quantity": inv["pending_po_quantity"],
            "net_need": net_need,
            "pack_size": pack,
            "packs": packs,
            "suggested_quantity": packs * pack,
        }

    def create_purchase_order(self, item_id: str, store_id: str, quantity: int) -> dict:
        now = _now()
        with self._write() as conn:
            row = self._item(conn, item_id, store_id)
            n = conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0] + 1
            po_id = f"PO-{n:04d}"
            conn.execute(
                "INSERT INTO purchase_orders VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (po_id, item_id, store_id, quantity, now, now),
            )
        return {
            "po_id": po_id,
            "status": "pending",
            "item_id": item_id,
            "name": row["name"],
            "store_id": store_id,
            "quantity": quantity,
            "message": "Order staged. It is not placed until the manager confirms it.",
        }

    # -- held-out tools ---------------------------------------------------------------------

    def get_promotions(self, store_id: str, days: int) -> dict:
        end = self.as_of + timedelta(days=days)
        with self._read() as conn:
            self._check_store(conn, store_id)
            rows = conn.execute(
                "SELECT p.promo_id, p.item_id, i.name, p.start_date, p.end_date, p.discount_pct, "
                "p.promo_price, pr.price AS regular_price FROM promotions p "
                "JOIN items i ON i.item_id = p.item_id "
                "LEFT JOIN prices pr ON pr.item_id = p.item_id AND pr.store_id = p.store_id "
                "WHERE p.store_id = ? AND p.start_date <= ? AND p.end_date > ? "
                "ORDER BY p.start_date, i.name",
                (store_id, end.isoformat(), self.as_of.isoformat()),
            ).fetchall()
        return {
            "store_id": store_id,
            "window_start": (self.as_of + timedelta(days=1)).isoformat(),
            "window_end": end.isoformat(),
            "promotions": [dict(r) for r in rows],
        }

    def request_stock_transfer(
        self, item_id: str, from_store: str, to_store: str, quantity: int
    ) -> dict:
        if from_store == to_store:
            raise ToolError("from_store and to_store must be different stores.")
        now = _now()
        with self._write() as conn:
            src = self._item(conn, item_id, from_store)
            self._item(conn, item_id, to_store)
            if quantity > src["on_hand"]:
                raise ToolError(
                    f"{from_store} has only {src['on_hand']} units of {item_id} on hand."
                )
            n = conn.execute("SELECT COUNT(*) FROM stock_transfers").fetchone()[0] + 1
            transfer_id = f"TR-{n:04d}"
            conn.execute(
                "INSERT INTO stock_transfers VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (transfer_id, item_id, from_store, to_store, quantity, now, now),
            )
        return {
            "transfer_id": transfer_id,
            "status": "pending",
            "item_id": item_id,
            "name": src["name"],
            "from_store": from_store,
            "to_store": to_store,
            "quantity": quantity,
            "message": "Transfer staged. It does not happen until the manager confirms it.",
        }

    # -- manager actions (gateway/UI only, never exposed to the model) ----------------------

    def confirm_purchase_order(self, po_id: str) -> dict:
        """Pending -> confirmed; the ordered units become on_order (in transit)."""
        with self._write() as conn:
            po = self._pending(conn, "purchase_orders", "po_id", po_id)
            conn.execute(
                "UPDATE purchase_orders SET status = 'confirmed', updated_at = ? WHERE po_id = ?",
                (_now(), po_id),
            )
            conn.execute(
                "UPDATE inventory SET on_order = on_order + ? WHERE item_id = ? AND store_id = ?",
                (po["quantity"], po["item_id"], po["store_id"]),
            )
        return {**dict(po), "status": "confirmed"}

    def cancel_purchase_order(self, po_id: str) -> dict:
        with self._write() as conn:
            po = self._pending(conn, "purchase_orders", "po_id", po_id)
            conn.execute(
                "UPDATE purchase_orders SET status = 'cancelled', updated_at = ? WHERE po_id = ?",
                (_now(), po_id),
            )
        return {**dict(po), "status": "cancelled"}

    def confirm_stock_transfer(self, transfer_id: str) -> dict:
        """Pending -> confirmed; units leave the source store and arrive as on_order at the target."""
        with self._write() as conn:
            tr = self._pending(conn, "stock_transfers", "transfer_id", transfer_id)
            src = self._item(conn, tr["item_id"], tr["from_store"])
            if tr["quantity"] > src["on_hand"]:
                raise ToolError(f"{tr['from_store']} no longer has {tr['quantity']} units on hand.")
            conn.execute(
                "UPDATE stock_transfers SET status = 'confirmed', updated_at = ? "
                "WHERE transfer_id = ?",
                (_now(), transfer_id),
            )
            conn.execute(
                "UPDATE inventory SET on_hand = on_hand - ? WHERE item_id = ? AND store_id = ?",
                (tr["quantity"], tr["item_id"], tr["from_store"]),
            )
            conn.execute(
                "UPDATE inventory SET on_order = on_order + ? WHERE item_id = ? AND store_id = ?",
                (tr["quantity"], tr["item_id"], tr["to_store"]),
            )
        return {**dict(tr), "status": "confirmed"}

    def cancel_stock_transfer(self, transfer_id: str) -> dict:
        with self._write() as conn:
            tr = self._pending(conn, "stock_transfers", "transfer_id", transfer_id)
            conn.execute(
                "UPDATE stock_transfers SET status = 'cancelled', updated_at = ? "
                "WHERE transfer_id = ?",
                (_now(), transfer_id),
            )
        return {**dict(tr), "status": "cancelled"}

    @staticmethod
    def _pending(conn, table: str, key: str, value: str) -> sqlite3.Row:
        row = conn.execute(f"SELECT * FROM {table} WHERE {key} = ?", (value,)).fetchone()
        if row is None:
            raise ToolError(f"No such record '{value}'.")
        if row["status"] != "pending":
            raise ToolError(f"'{value}' is already {row['status']}.")
        return row


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
