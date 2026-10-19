"""Minimal client for the teacher model: any OpenAI-compatible chat completions endpoint.

Configured from the environment (see `.env.example`): TEACHER_BASE_URL, TEACHER_API_KEY,
TEACHER_MODEL.
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx
from dotenv import load_dotenv


class TeacherNotConfigured(RuntimeError):
    pass


class Teacher:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        load_dotenv()
        self.base_url = (base_url or os.getenv("TEACHER_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("TEACHER_API_KEY", "")
        self.model = model or os.getenv("TEACHER_MODEL", "")
        if not self.base_url or "example.com" in self.base_url or not self.api_key:
            raise TeacherNotConfigured("Set TEACHER_BASE_URL and TEACHER_API_KEY in .env")
        if not self.model:
            raise TeacherNotConfigured("Set TEACHER_MODEL in .env")
        # Short connect timeout: over the VPN a stalled connection is better retried quickly.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def list_models(self) -> list[str]:
        resp = self._client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retries: int = 3,
        **extra,
    ) -> dict:
        """Return the assistant message dict (content and optional tool_calls)."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        if tools:
            body["tools"] = tools
        for attempt in range(retries + 1):
            try:
                resp = self._client.post(f"{self.base_url}/chat/completions", json=body)
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
                    break
            except httpx.TransportError:
                if attempt == retries:
                    raise
            time.sleep(2 * 2**attempt)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and surrounding prose."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])
