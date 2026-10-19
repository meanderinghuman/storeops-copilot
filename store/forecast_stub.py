"""Forecast service stub with the ShelfSense API contract.

    uv run uvicorn store.forecast_stub:app --port 8001

POST /forecast {"item_id": "FOODS_3_090", "store_id": "CA_1", "horizon_days": 7}
-> {"item_id", "store_id", "horizon_days", "start_date", "end_date", "model",
    "total": {"p10", "p50", "p90"}, "daily": [{"date", "p10", "p50", "p90"}, ...]}

Swapping in ShelfSense later means pointing FORECAST_URL at it; the request and response
format stay the same.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from store.forecast import MAX_HORIZON_DAYS, MODEL_NAME, UnknownSeries, seasonal_naive

load_dotenv()


class ForecastRequest(BaseModel):
    item_id: str
    store_id: str
    horizon_days: int = Field(7, ge=1, le=MAX_HORIZON_DAYS)


class Quantiles(BaseModel):
    p10: float
    p50: float
    p90: float


class DailyQuantiles(Quantiles):
    date: str


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    horizon_days: int
    start_date: str
    end_date: str
    model: str
    total: Quantiles
    daily: list[DailyQuantiles]


def create_app(db_path: str | Path | None = None) -> FastAPI:
    db = Path(db_path or os.getenv("STORE_DB_PATH", "data/store.db"))
    app = FastAPI(title="Forecast stub (ShelfSense contract)")

    def connect() -> sqlite3.Connection:
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": MODEL_NAME, "db": str(db), "db_exists": db.exists()}

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest) -> dict:
        conn = connect()
        try:
            return seasonal_naive(conn, req.item_id, req.store_id, req.horizon_days)
        except UnknownSeries as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            conn.close()

    return app


app = create_app()
