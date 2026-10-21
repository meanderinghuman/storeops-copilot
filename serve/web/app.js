// StoreOps Copilot web UI. Plain JS, no build step; talks to the gateway on the same origin.
"use strict";

// Display names are fictional labels for the M5 dataset's anonymized California stores.
const STORES = [
  { id: "CA_1", name: "Sacramento Midtown" },
  { id: "CA_2", name: "San Jose North" },
  { id: "CA_3", name: "Fresno Central" },
  { id: "CA_4", name: "Riverside Plaza" },
];
const storeName = (id) => (STORES.find((s) => s.id === id) || { name: id }).name;
// Each example was checked to be answered locally 3/3 times (others work too, via escalation).
const EXAMPLES = [
  "Will we run out of Fresh bananas bunch 2 lb before the weekend?",
  "Should I order more Toilet paper 12-roll pack?",
  "Order 24 Dish soap liquid 20 oz.",
  "How many Paper towels 6-pack do we have?",
  "Order more paper towels.",
  "Any promotions coming up in the next 7 days?",
  "What's the weather tomorrow?",
];
const TOOL_LABEL = {
  search_items: "Searched the catalog",
  get_inventory: "Checked stock",
  get_sales_history: "Pulled sales history",
  get_forecast: "Forecast demand",
  get_reorder_suggestion: "Calculated reorder quantity",
  create_purchase_order: "Drafted a purchase order",
  get_promotions: "Checked promotions",
  request_stock_transfer: "Drafted a stock transfer",
};
const REASON = {
  refused_call: "local model skipped a required step",
  tool_error: "a tool call failed",
  ungrounded_number: "local answer had an unverified number",
  max_rounds: "too many tool rounds",
  timeout: "time limit reached",
  bad_output: "malformed local output",
  local_error: "local model unavailable",
};

const state = {
  store: ["CA_1", "CA_2", "CA_3", "CA_4"].includes(localStorage.getItem("store")) ? localStorage.getItem("store") : "CA_3",
  chats: {},            // store -> [{message, answer}]
  stats: { n: 0, local: 0, seconds: 0 },
  apiKey: sessionStorage.getItem("apiKey") || "",
};
const $ = (id) => document.getElementById(id);

// ---- helpers ------------------------------------------------------------------------------

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal markdown on already-escaped text: **bold**, `code`, bullet/numbered lists, paragraphs.
function md(text) {
  const blocks = esc(text).split(/\n{2,}/);
  return blocks.map((b) => {
    const lines = b.split("\n");
    const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
    if (lines.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l))) {
      return "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*([-*•]|\d+[.)])\s+/, ""))}</li>`).join("") + "</ul>";
    }
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  }).join("");
}

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

const fmt = (n) => (Number.isInteger(n) ? n.toLocaleString() : Number(n).toFixed(1));
// "2016-05-23" -> "Mon, May 23" (the data's calendar; the year is left out on purpose)
function day(iso, weekday = true) {
  const d = new Date(`${iso}T12:00:00Z`);
  return d.toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric", ...(weekday ? { weekday: "short" } : {}) });
}

async function api(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  const r = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
  if (r.status === 401) {
    const key = prompt("This gateway needs an API key (X-API-Key):");
    if (key) { state.apiKey = key; sessionStorage.setItem("apiKey", key); return api(method, path, body); }
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : r.statusText);
  return data;
}

// ---- tool step renderers ------------------------------------------------------------------

function argsText(name, a) {
  if (!a) return "";
  if (name === "search_items") return `“${a.query}”`;
  const parts = [a.item_id, a.from_store && `${a.from_store} → ${a.to_store}`, a.quantity && `qty ${a.quantity}`,
    a.days && `${a.days} days`, a.weeks && `${a.weeks} weeks`, a.service_level && `service ${a.service_level}`];
  return parts.filter(Boolean).join(" · ");
}

function tiles(items) {
  return `<div class="tiles">${items.map(([k, v, hl]) =>
    `<div class="tile${hl ? " hl" : ""}"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("")}</div>`;
}

const RENDER = {
  search_items: (r) => r.matches.length
    ? `<div class="chips">${r.matches.map((m) => `<span class="chip">${esc(m.name)}<span class="id">${esc(m.item_id)}</span></span>`).join("")}</div>`
    : `<div class="muted">No matching products.</div>`,
  get_inventory: (r) => tiles([["on hand", fmt(r.on_hand), true], ["on order", fmt(r.on_order)],
    ["pending POs", fmt(r.pending_po_quantity)], ["lead time", `${r.lead_time_days} d`], ["pack size", fmt(r.pack_size)]]),
  get_sales_history: (r) => {
    const max = Math.max(1, ...r.weeks.map((w) => w.units));
    const bars = r.weeks.map((w) => `<div title="${esc(w.week_start)} – ${esc(w.week_end)}: ${w.units}">
      <span>${w.units}</span><i style="height:${Math.max(3, (w.units / max) * 54)}px"></i><span>${esc(day(w.week_start, false))}</span></div>`).join("");
    return `<div class="bars">${bars}</div><div class="muted" style="font-size:12px">total ${fmt(r.total_units)} · ${fmt(r.avg_units_per_week)}/week</div>`;
  },
  get_forecast: (r) => {
    const t = r.total, max = Math.max(1, t.p90 * 1.15), pos = (v) => `${(v / max) * 100}%`;
    return `<div class="range"><div class="track"></div>
      <div class="band" style="left:${pos(t.p10)};width:calc(${pos(t.p90)} - ${pos(t.p10)})"></div>
      <div class="mid" style="left:${pos(t.p50)}"></div>
      <span class="lbl" style="left:${pos(t.p10)}">low ${fmt(t.p10)}</span>
      <span class="lbl m" style="left:${pos(t.p50)}">expected ${fmt(t.p50)}</span>
      <span class="lbl" style="left:${pos(t.p90)}">high ${fmt(t.p90)}</span></div>
      <div class="muted" style="font-size:12px">${esc(day(r.start_date))} – ${esc(day(r.end_date))} · ${r.horizon_days} days</div>`;
  },
  get_reorder_suggestion: (r) => tiles([["suggested order", fmt(r.suggested_quantity), true],
    [`demand @ ${Math.round(r.service_level * 100)}%`, fmt(r.demand_at_service_level)], ["on hand", fmt(r.on_hand)],
    ["on order", fmt(r.on_order + r.pending_po_quantity)], ["covers", `${r.coverage_days} d`], ["pack", fmt(r.pack_size)]]),
  get_promotions: (r) => r.promotions.length
    ? r.promotions.slice(0, 8).map((p) => `<div class="promo"><span>${esc(p.name)}</span>
        <span class="muted">${p.discount_pct}% off · ${esc(day(p.start_date, false))} – ${esc(day(p.end_date, false))}</span></div>`).join("")
      + (r.promotions.length > 8 ? `<div class="muted" style="font-size:12px">+${r.promotions.length - 8} more</div>` : "")
    : `<div class="muted">No promotions in this window.</div>`,
  create_purchase_order: (r) => `<div class="muted">${esc(r.po_id)} · ${fmt(r.quantity)} × ${esc(r.name)} · pending your approval</div>`,
  request_stock_transfer: (r) => `<div class="muted">${esc(r.transfer_id)} · ${fmt(r.quantity)} × ${esc(r.name)} · ${esc(r.from_store)} → ${esc(r.to_store)} · pending</div>`,
};

function renderStep(c) {
  let args = null;
  try { args = typeof c.arguments === "string" ? JSON.parse(c.arguments) : c.arguments; } catch { /* shown raw */ }
  const title = TOOL_LABEL[c.name] || c.name;
  let body;
  if (!c.ok) body = `<div class="refused">Refused: ${esc(c.error)}</div>`;
  else {
    try { body = (RENDER[c.name] || (() => ""))(c.result); } catch { body = ""; }
  }
  return `<div class="step${c.ok ? "" : " err"}"><div class="step-title">${esc(title)}
    <span class="args">${esc(args ? argsText(c.name, args) : c.arguments)}</span></div>
    <div class="step-body">${body}</div></div>`;
}

// ---- approvals ----------------------------------------------------------------------------

function approvalCard(p, inline = false) {
  const isPo = Boolean(p.po_id);
  const ref = p.po_id || p.transfer_id;
  const what = isPo ? "Purchase order" : `Transfer ${storeName(p.from_store)} → ${storeName(p.to_store)}`;
  return `<div class="card${inline ? " inline" : ""}" data-ref="${esc(ref)}" data-kind="${isPo ? "orders" : "transfers"}">
    <div class="card-head"><span class="card-id">${esc(ref)}</span><span class="card-qty">${esc(what)}</span></div>
    <div class="card-title">${fmt(p.quantity)} × ${esc(p.name || p.item_id)}</div>
    <div class="card-actions">
      <button class="btn primary" data-act="confirm">Confirm</button>
      <button class="btn" data-act="cancel">Cancel</button>
    </div></div>`;
}

async function act(card, action) {
  const { ref, kind } = card.dataset;
  card.querySelectorAll("button").forEach((b) => (b.disabled = true));
  try {
    await api("POST", `/v1/${kind}/${encodeURIComponent(ref)}/${action}`);
    document.querySelectorAll(`.card[data-ref="${CSS.escape(ref)}"] .card-actions`).forEach((a) => {
      a.outerHTML = `<div class="state ${action === "confirm" ? "confirmed" : "cancelled"}">${action === "confirm" ? "✓ Confirmed — order placed" : "Cancelled"}</div>`;
    });
  } catch (e) {
    alert(e.message);
    card.querySelectorAll("button").forEach((b) => (b.disabled = false));
  }
  refreshApprovals();
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-act]");
  if (b) act(b.closest(".card"), b.dataset.act);
});

async function refreshApprovals() {
  try {
    const [orders, transfers] = await Promise.all([
      api("GET", `/v1/orders?store_id=${state.store}`), api("GET", `/v1/transfers?store_id=${state.store}`)]);
    const all = [...orders, ...transfers];
    $("approval-count").textContent = all.length;
    $("approval-count").classList.toggle("zero", all.length === 0);
    $("approvals").innerHTML = all.length ? all.map((p) => approvalCard(p)).join("") : `<div class="empty">Nothing pending</div>`;
  } catch { /* gateway down: status panel shows it */ }
}

async function refreshStatus() {
  const set = (id, cls, text) => { $(id).className = `dot ${cls}`; $(`${id}-text`).textContent = text; };
  let checks;
  try {
    const r = await fetch("/ready");
    checks = (await r.json());
    checks = checks.detail || checks;
  } catch { set("st-llm", "bad", "offline"); set("st-esc", "bad", "offline"); set("st-db", "bad", "offline"); return; }
  set("st-llm", checks.local_llm === "ok" ? "ok" : "bad", checks.local_llm === "ok" ? "online" : "offline");
  set("st-esc", checks.escalation === "configured" ? "ok" : "warn", checks.escalation === "configured" ? "configured" : "off");
  set("st-db", checks.store_db === "ok" ? "ok" : "bad", checks.store_db === "ok" ? "ok" : "error");
}

// ---- chat -------------------------------------------------------------------------------

function updateStats() {
  const s = state.stats;
  $("s-req").textContent = s.n;
  $("s-local").textContent = s.n ? `${Math.round((100 * s.local) / s.n)}%` : "–";
  $("s-lat").textContent = s.n ? `${(s.seconds / s.n).toFixed(1)} s` : "–";
}

function scrollDown() { $("thread").scrollTop = $("thread").scrollHeight; }

function addUser(text) {
  $("welcome")?.remove();
  $("thread").appendChild(el(`<div class="msg user"><div class="bubble">${esc(text)}</div></div>`));
  scrollDown();
}

const AVATAR = `<div class="avatar"><svg viewBox="0 0 24 24"><path d="M4 9h16l-2 10H6z"/><path d="M8.5 9a3.5 3.5 0 0 1 7 0"/></svg></div>`;

function addThinking() {
  const node = el(`<div class="msg">${AVATAR}<div class="assistant-body"><div class="answer">
    <span class="typing"><span></span><span></span><span></span></span><span class="typing-text">Checking the store systems…</span></div></div></div>`);
  $("thread").appendChild(node);
  scrollDown();
  return node;
}

function renderAssistant(node, out) {
  const route = out.route;
  const reasons = (out.escalation_reasons || []).map((r) => REASON[r.split(":")[0]] || r);
  const badge = route === "local"
    ? `<span class="badge local"><svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8"/></svg>Answered locally</span>`
    : route === "escalated"
      ? `<span class="badge escalated"><svg viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>Escalated to large model</span>`
      : `<span class="badge fallback">Couldn't complete</span>`;
  const why = route !== "local" && reasons.length ? `<span>${esc(reasons.join(", "))}</span>` : "";
  const steps = out.trace || [];
  const pending = (out.pending || []).map((p) => approvalCard(p, true)).join("");
  node.querySelector(".assistant-body").innerHTML = `
    <div class="answer">${md(out.answer)}${pending}</div>
    <div class="meta">${badge}${why}<span>${out.seconds.toFixed(1)} s</span>
      ${steps.length ? `<button class="toggle">Show how (${steps.length} step${steps.length > 1 ? "s" : ""})</button>`
        : route === "fallback" ? `<span>the local answer was rejected and the large model isn't reachable</span>` : `<span>no tools needed</span>`}</div>
    <div class="trace" hidden>${steps.map(renderStep).join("")}</div>`;
  const toggle = node.querySelector(".toggle");
  if (toggle) toggle.onclick = () => {
    const t = node.querySelector(".trace");
    t.hidden = !t.hidden;
    toggle.textContent = t.hidden ? `Show how (${steps.length} step${steps.length > 1 ? "s" : ""})` : "Hide steps";
  };
  scrollDown();
}

// "yes, go ahead", "make it 36 instead", "what about the 12-pack?", "no, hold off" ...
function isFollowUp(text) {
  const t = text.trim().toLowerCase();
  return /^(yes|yeah|yep|yup|ok|okay|sure|no|nope|not yet|go ahead|do it|please do|make it|let'?s do|what about|how about|and |also |same )/.test(t)
    || /\b(instead|that one|those|them|it)\b/.test(t) && t.split(/\s+/).length <= 8;
}

async function send(text) {
  text = text.trim();
  if (!text) return;
  const turns = (state.chats[state.store] ||= []);
  // The model was trained on single requests, and unrelated earlier turns measurably change its
  // behavior. So the last exchange goes along only when this message is a follow-up to it.
  const history = isFollowUp(text) && turns.length ? turns[turns.length - 1].turn : [];
  addUser(text);
  const node = addThinking();
  $("send").disabled = true;
  try {
    const out = await api("POST", "/v1/chat", { store_id: state.store, message: text, history });
    renderAssistant(node, out);
    turns.push({ message: text, answer: out.answer, turn: out.turn || [] });
    Object.assign(state.stats, { n: state.stats.n + 1, local: state.stats.local + (out.route === "local"), seconds: state.stats.seconds + out.seconds });
    updateStats();
    if ((out.pending || []).length) refreshApprovals();
  } catch (e) {
    node.querySelector(".assistant-body").innerHTML = `<div class="answer"><div class="refused">${esc(e.message)}</div></div>`;
  } finally {
    $("send").disabled = false;
    $("input").focus();
  }
}

// ---- layout -----------------------------------------------------------------------------

function selectStore(store) {
  state.store = store;
  localStorage.setItem("store", store);
  $("store-title").textContent = storeName(store);
  $("store-id").textContent = `Store ${store}`;
  document.querySelectorAll("#stores button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.store === store)));
  refreshApprovals();
}

function init() {
  $("stores").innerHTML = STORES.map((s) => `<button role="tab" data-store="${s.id}">
    <span class="store-name">${esc(s.name)}</span><span class="store-id">${s.id}</span></button>`).join("");
  $("stores").onclick = (e) => { const b = e.target.closest("button"); if (b) selectStore(b.dataset.store); };
  $("examples").innerHTML = EXAMPLES.map((x) => `<button>${esc(x)}</button>`).join("");
  $("examples").onclick = (e) => { const b = e.target.closest("button"); if (b) send(b.textContent); };
  const input = $("input");
  const grow = () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 160)}px`; };
  input.addEventListener("input", grow);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  $("composer").onsubmit = (e) => { e.preventDefault(); const t = input.value; input.value = ""; grow(); send(t); };
  $("new-chat").onclick = () => { state.chats[state.store] = []; location.reload(); };
  selectStore(state.store);
  refreshStatus();
  setInterval(refreshStatus, 15000);
  setInterval(refreshApprovals, 5000);
  updateStats();
  input.focus();
}

init();
