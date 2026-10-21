"""StoreOps Copilot chat UI (Streamlit).

    uv run --group ui streamlit run serve/ui/app.py

Talks to the gateway only: chat, the tool trace of each answer, and Confirm / Cancel buttons for
pending orders and transfers (the only way anything gets confirmed).
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

GATEWAY = os.getenv("GATEWAY_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": os.getenv("GATEWAY_API_KEY", "")} if os.getenv("GATEWAY_API_KEY") else {}
STORES = ["CA_1", "CA_2", "CA_3", "CA_4"]
EXAMPLES = [
    "How many Paper towels 6-pack do we have?",
    "Will we run out of Fresh bananas bunch 2 lb before the weekend?",
    "Order 24 Dish soap liquid 20 oz.",
    "Any promotions coming up in the next 7 days?",
    "What's the weather tomorrow?",
]

st.set_page_config(page_title="StoreOps Copilot", page_icon="🛒", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{GATEWAY}{path}", headers=HEADERS, timeout=120, **kw)
    except httpx.HTTPError as exc:
        st.error(f"Gateway unreachable at {GATEWAY}: {type(exc).__name__}")
        return None
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.text[:300]}")
        return None
    return r.json()


if "chats" not in st.session_state:
    st.session_state.chats = {}  # store_id -> list of turns

with st.sidebar:
    st.title("🛒 StoreOps Copilot")
    store = st.selectbox("Store", STORES, index=2)
    st.caption("Answers come from a fine-tuned 1.5B model running locally; requests it can't "
               "handle reliably are escalated to a large model. Orders are never placed without "
               "your confirmation.")  # fmt: skip
    st.subheader("Try")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_prompt = ex
    if st.button("New conversation", type="secondary"):
        st.session_state.chats[store] = []
        st.rerun()

    st.subheader("Pending orders")
    orders = api("GET", "/v1/orders", params={"store_id": store}) or []
    if not orders:
        st.caption("None")
    for o in orders:
        with st.container(border=True):
            st.write(f"**{o['po_id']}** · {o['item_id']} · {o['quantity']} units")
            c1, c2 = st.columns(2)
            if c1.button("Confirm", key=f"c{o['po_id']}", type="primary"):
                api("POST", f"/v1/orders/{o['po_id']}/confirm")
                st.rerun()
            if c2.button("Cancel", key=f"x{o['po_id']}"):
                api("POST", f"/v1/orders/{o['po_id']}/cancel")
                st.rerun()

turns = st.session_state.chats.setdefault(store, [])


def show_turn(t: dict) -> None:
    with st.chat_message("user"):
        st.write(t["message"])
    with st.chat_message("assistant"):
        st.write(t["answer"])
        route = t.get("route", "local")
        badge = {"local": "🟢 answered locally", "escalated": "🟠 escalated to large model",
                 "fallback": "🔴 fallback"}[route]  # fmt: skip
        reasons = f" ({', '.join(t['escalation_reasons'])})" if t.get("escalation_reasons") else ""
        st.caption(f"{badge}{reasons} · {t.get('seconds', 0):.1f} s")
        for p in t.get("pending", []):
            ref = p.get("po_id") or p.get("transfer_id")
            st.info(f"Pending {ref}: {p['quantity']} × {p.get('name', p['item_id'])}. "
                    "Confirm it in the sidebar.")  # fmt: skip
        if t.get("trace"):
            with st.expander(f"Tool calls ({len(t['trace'])})"):
                for c in t["trace"]:
                    icon = "✅" if c["ok"] else "⛔"
                    st.markdown(f"{icon} `{c['name']}` `{c['arguments']}`")
                    st.code(json.dumps(c["result"] if c["ok"] else {"error": c["error"]}, indent=1)[:1500],
                            language="json")  # fmt: skip


st.header(f"Store {store}")
for t in turns:
    show_turn(t)

prompt = st.chat_input("Ask about stock, sales, forecasts, or ordering…")
prompt = prompt or st.session_state.pop("pending_prompt", None)
if prompt:
    history = []
    for t in turns[-6:]:  # earlier turns as plain text; the model re-checks tools as needed
        history += [
            {"role": "user", "content": t["message"]},
            {"role": "assistant", "content": t["answer"]},
        ]
    with st.spinner("Thinking…"):
        out = api(
            "POST", "/v1/chat", json={"store_id": store, "message": prompt, "history": history}
        )
    if out:
        turns.append({"message": prompt, **out})
        st.rerun()
