"""Write-intent check: an order or transfer must match what the manager actually asked for.

Rules (deterministic, no model):
- A purchase order needs ordering language in the manager's message ("order", "PO", "restock",
  "buy", ...), or a reply to an order the assistant just offered ("yes, go ahead").
- A request to send/move stock to or from another store must be a transfer, not a purchase order.
- A transfer needs moving language ("send", "move", "transfer", ...).

A quantity rule ("use the number the manager typed") was tried and dropped: managers paraphrase
product names ("2-quart containers", "12pk", "9x13 dish"), so sizes were misread as quantities and
it blocked 2% of legitimate training orders while catching 1 of 13 wrong writes.
`stated_quantities` is kept for analysis.
"""

from __future__ import annotations

import re

ORDER = re.compile(
    r"\b(order|orders|ordering|ordered|reorder|re-order|po|p\.o\.|purchase|stage|buy|restock|"
    r"re-stock|stock up|get (?:some|more|in)|bring in|put in)\b",
    re.IGNORECASE,
)
_MOVE_WORDS = (
    r"send|sent|sending|move[ds]?|moving|transfer(?:red|ring|s)?|ship(?:ped|ping|s)?|"
    r"pull(?:ed)?|bring|brought|get"
)
MOVE = re.compile(rf"\b({_MOVE_WORDS})\b", re.IGNORECASE)
TO_OTHER_STORE = re.compile(
    rf"\b({_MOVE_WORDS})\b[^.?!]*\b(to|from)\b[^.?!]*\b[A-Z]{{2}}_\d+\b", re.IGNORECASE
)
YES = re.compile(
    r"^\W*(yes|yeah|yep|ok|okay|sure|go ahead|do it|please do|sounds good|make it|let'?s do)\b",
    re.IGNORECASE,
)
UNITS = (
    r"(?:fl\s*oz|oz|lbs?|ft|in|inch(?:es)?|qt|gal(?:lon)?s?|l|ml|liters?|count|ct|pack|pcs?|"
    r"pieces?|sq\s*ft|rolls?|hooks|tier)"
)
SIZE = re.compile(rf"(?<![\d.])\d+(?:\.\d+)?\s*-?\s*{UNITS}\b", re.IGNORECASE)
TIME_SPAN = re.compile(r"\b\d+\s*-?\s*(?:days?|weeks?|wks?|months?|hours?|hrs?)\b", re.IGNORECASE)
NOT_QTY = re.compile(r"\b\d+\s+(?:left|remaining|on hand|in stock|on the shelf)\b", re.IGNORECASE)
STORE = re.compile(r"\b[A-Z]{2}_\d+\b")


def stated_quantities(message: str, item_name: str) -> set[int]:
    """Whole numbers the manager typed as quantities."""
    text = re.sub(re.escape(item_name), " ", message, flags=re.IGNORECASE)
    for pattern in (STORE, TIME_SPAN, NOT_QTY, SIZE):
        text = pattern.sub(" ", text)
    return {int(n) for n in re.findall(r"(?<![\d.])\d+(?![\d.])", text)}


def check_write(
    tool: str, quantity: int | None, item_name: str, message: str, last_assistant: str
) -> str | None:
    """Refusal message if the write doesn't match the manager's request, else None."""
    offered = bool(
        re.search(r"\b(stage|order|transfer|move|send)\b", last_assistant or "", re.IGNORECASE)
    )
    confirming = offered and bool(YES.search(message))
    if tool == "create_purchase_order":
        if TO_OTHER_STORE.search(message):
            return (
                "The manager asked to move stock between stores. Use request_stock_transfer, "
                "not a purchase order."
            )
        if not ORDER.search(message) and not confirming:
            return (
                "The manager did not ask to place an order. Answer the question, and offer to "
                "stage an order instead of staging one."
            )
    if tool == "request_stock_transfer" and not MOVE.search(message) and not confirming:
        return "The manager did not ask to move stock between stores."
    return None
