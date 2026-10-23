"""Gateway settings, all from the environment (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _url(value: str) -> str:
    """The .env.example placeholder means 'not configured' (local-only mode)."""
    return "" if "example.com" in value else value


@dataclass(frozen=True)
class Settings:
    # Local model: llama-server (OpenAI-compatible)
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "storeops-3b-q4_k_m"
    # Escalation: large model behind an OpenAI-compatible API (the teacher endpoint by default)
    escalation_enabled: bool = True
    escalation_base_url: str = ""
    escalation_api_key: str = ""
    escalation_model: str = ""
    # Store backend
    db_path: str = "data/store.db"
    forecast_url: str | None = None
    # Safety limits
    max_rounds: int = 4
    request_timeout_s: float = 45.0
    qty_cap_p90_multiple: float = 5.0  # order cap = max(this x p90 demand over coverage, N packs)
    qty_cap_min_packs: int = 4
    # API protection
    api_key: str = ""  # if set, /v1/* requires header X-API-Key
    rate_limit_per_min: int = 60
    tool_names: tuple[str, ...] = field(default=())

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        e = os.getenv
        return cls(
            llm_base_url=e("LLM_BASE_URL", cls.llm_base_url),
            llm_model=e("LLM_MODEL", cls.llm_model),
            escalation_enabled=_bool("ESCALATION_ENABLED", True),
            escalation_base_url=_url(e("ESCALATION_BASE_URL") or e("TEACHER_BASE_URL", "")),
            escalation_api_key=e("ESCALATION_API_KEY") or e("TEACHER_API_KEY", ""),
            escalation_model=e("ESCALATION_MODEL") or e("TEACHER_MODEL", ""),
            db_path=e("STORE_DB_PATH", cls.db_path),
            forecast_url=e("FORECAST_URL") or None,
            max_rounds=int(e("MAX_TOOL_ROUNDS", cls.max_rounds)),
            request_timeout_s=float(e("REQUEST_TIMEOUT_S", cls.request_timeout_s)),
            api_key=e("GATEWAY_API_KEY", ""),
            rate_limit_per_min=int(e("RATE_LIMIT_PER_MIN", cls.rate_limit_per_min)),
        )
