"""Prometheus metrics exported by the gateway at /metrics."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("copilot_requests_total", "Chat requests", ["route", "outcome"])
LATENCY = Histogram(
    "copilot_request_seconds",
    "End-to-end latency per chat request",
    ["route"],
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 20, 30, 45, 60),
)
TOOL_CALLS = Counter(
    "copilot_tool_calls_total", "Tool calls proposed by a model", ["tool", "status"]
)
REJECTED = Counter("copilot_rejected_calls_total", "Tool calls refused or failed", ["reason"])
ESCALATIONS = Counter(
    "copilot_escalations_total", "Requests escalated to the large model", ["reason"]
)
RETRIES = Counter("copilot_invalid_output_retries_total", "Retries after malformed model output")
ORDERS = Counter("copilot_orders_total", "Purchase orders and transfers", ["kind", "action"])
TOKENS = Counter("copilot_tokens_total", "LLM tokens", ["route", "kind"])
