<div align="center">

# 🛒 StoreOps Copilot

**A fine-tuned 3B model that answers store managers' inventory questions by calling store tools,
running locally on a MacBook, with a safety gateway that escalates the hardest requests to a large model.**

[![CI](https://github.com/meanderinghuman/storeops-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/meanderinghuman/storeops-copilot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-storeops--copilot--qwen2.5--3b-FFD21E)](https://huggingface.co/sidpnyc/storeops-copilot-qwen2.5-3b)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Apple%20Metal-000000?logo=apple&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-gateway-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800?logo=grafana&logoColor=white)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)

![Task success](https://img.shields.io/badge/task%20success-95.4%25-brightgreen)
![Answered locally](https://img.shields.io/badge/answered%20locally-94%25-brightgreen)
![Harmful actions](https://img.shields.io/badge/harmful%20actions-0.4%25-brightgreen)
![Cost](https://img.shields.io/badge/API%20cost-%E2%88%9294%25-brightgreen)

<img src="docs/demo.gif" width="900" alt="Demo: a run-out question answered locally from stock and forecast tools, 'yes, go ahead' staging a purchase order that the manager confirms, a two-product comparison escalated to the large model, and an out-of-scope question declined">

</div>

> *"Will we run out of bananas before the weekend?"* → the copilot searches the catalog, checks stock
> and the demand forecast, computes a reorder quantity in code, and offers to stage a purchase order.
> *"Yes, go ahead"* → a pending order appears, and nothing is placed until the manager clicks **Confirm**.

## At a glance

| | |
| --- | --- |
| 🎯 **95.4% task success** on 500 held-out scenarios | stores and tools never seen in training |
| 💻 **94% of requests answered locally**, for $0 | fine-tuned Qwen2.5-3B on Apple Metal |
| 🛡️ **0.4% harmful actions**, half the large model's rate | safety rules enforced in code, not in the prompt |
| ⚡ **3.7 s median** per conversation | on par with sending everything to the API (3.8 s) |
| 💸 **94% lower API cost** | only the hardest 6% goes to the large model |
| 📈 **38% → 90%** from fine-tuning alone | execution-verified training data, QLoRA on a free T4 |

---

## How it works

Follow one request through the system:

```mermaid
sequenceDiagram
    autonumber
    actor M as Store manager
    participant UI as Web UI
    participant G as Gateway
    participant L as Local model<br/>(Qwen2.5-3B, Metal)
    participant T as Store tools
    participant B as Large model API
    M->>UI: "Will we run out of bananas before the weekend?"
    UI->>G: POST /v1/chat
    G->>L: question + system prompt + tool schemas
    L-->>G: call search_items("bananas")
    G->>G: guards: valid schema? real IDs? allowed write?
    G->>T: run the tool
    T-->>G: matching products
    L-->>G: call get_inventory, get_forecast, get_reorder_suggestion
    G->>T: run the tools
    T-->>G: stock 8, forecast 5.2 (high 10.3), suggest 6
    L-->>G: final answer
    G->>G: every number grounded in a tool result?
    alt answer can't be trusted (6% of requests)
        G->>B: re-run the request on the large model
        B-->>G: answer
    end
    G-->>UI: answer + tool trace + route (local / escalated)
    M->>UI: "yes, go ahead" → order PO-0001 is staged as pending
    M->>UI: clicks Confirm
    UI->>G: POST /v1/orders/PO-0001/confirm (the model can't call this)
```

**Step by step:**

1. **The manager asks a question** in the web UI, in plain English (typos are fine).
2. **The gateway sends it to the local model** with the system prompt and the store tools' JSON
   Schemas. The model runs natively on the Mac's GPU through `llama-server`.
3. **The model proposes tool calls**: search the catalog, check stock, pull sales, forecast demand,
   get a reorder suggestion, draft an order or a transfer between stores. It never runs anything itself.
4. **The gateway checks every call** before running it: the arguments must match the schema, the
   product ID must come from a search in this conversation (no invented IDs), an order must match
   what the manager asked for, and quantities must be reasonable. A refused call goes back to the
   model with the reason, and the model usually corrects itself.
5. **The tools run against the store database** (real Walmart M5 sales data) and the forecast
   service. Reorder quantities are calculated in code, so the model never does arithmetic on orders.
6. **The gateway checks the answer.** Every number must appear in a tool result. If anything looks
   wrong (an unrecovered refusal or error, a made-up number, too many steps), anything the local
   attempt drafted is cancelled and the request is re-run on a large model API. 94% of requests
   never need it.
7. **Orders are only ever drafts.** They wait in an approval queue; only the manager's **Confirm**
   button places them. The model has no tool that can confirm anything.
8. **Everything is observable:** each answer shows its steps and route in the UI, and Prometheus +
   Grafana track latency, local vs escalated share, refusals and the order lifecycle.

<p align="center"><img src="docs/screens/chat_dark.png" width="820" alt="A sales question answered with a weekly chart, then a transfer request where the gateway refuses an invented item ID, the model searches the catalog, and stages the right transfer, which the manager confirms"></p>
<p align="center"><sub>Above: the gateway refuses an invented product ID, the model searches and stages the right transfer, and the manager confirms. <a href="docs/screens/chat_light.png">Light mode</a> · <a href="docs/screens/welcome_light.png">welcome screen</a>. Store names are fictional labels for the dataset's anonymized California stores.</sub></p>

## How it was built

```mermaid
flowchart LR
  A[1 · Store backend<br/>M5 sales → SQLite<br/>8 tools + schemas] --> B[2 · Training data<br/>teacher answers 2,797<br/>questions with real<br/>tool execution]
  B --> C[3 · Filters<br/>2,645 kept<br/>94.6%]
  C --> D[4 · QLoRA fine-tune<br/>Qwen2.5 1.5B and 3B<br/>free Kaggle T4]
  D --> E[5 · Evaluate<br/>500 held-out<br/>scenarios]
  E --> F[6 · Serve<br/>llama.cpp on Metal<br/>+ safety gateway]
```

1. **Store backend** ([`store/`](store/)): real daily sales from the
   [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) Walmart dataset
   (California, 3 departments, 300 products, 4 stores, 2 of them kept out of training entirely) plus
   generated product names, inventory, lead times and promotions. Eight tools with JSON Schemas and
   a forecast service that returns low / expected / high demand.
2. **Execution-verified training data** ([`datagen/`](datagen/)): one large model (GLM-5.3) wrote
   2,797 realistic manager questions; another (DeepSeek-V4.1-Flash) answered them as an agent, with
   **every tool call executed for real** against the store database.
3. **Automatic filters** keep a conversation only if every number in the answer comes from a tool
   result, orders are staged only when asked (and end with a confirmation request), vague requests
   get a clarifying question and off-topic ones get no tool call: 2,645 kept (94.6%).
4. **Fine-tuning** ([`train/`](train/)): QLoRA with Unsloth on a free Kaggle T4, loss on the
   assistant's turns only, 2 epochs, plus 500 general function-calling examples from
   [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k). Same data and recipe
   for Qwen2.5-1.5B (2.6 h) and Qwen2.5-3B (4.7 h).
5. **Evaluation** ([`eval/`](eval/)): [500 scenarios](eval/TESTSET.md) frozen in git before
   training, at stores and with two tools the model never saw, plus a hard split (typos, two products
   in one message, follow-ups). A scripted perfect agent scores 100%, which checks the scorer itself.
6. **Serving** ([`serve/`](serve/)): `llama-server` on the Mac's GPU, a FastAPI gateway with the
   safety rules and escalation, the web UI, and Prometheus + Grafana, all started by `make up`.

## Results

500 held-out scenarios, temperature 0, the same prompt and tools for every model.

| System | Task success (95% CI) | Answered locally | Harmful actions¹ | p50 / p95 latency | API cost / 1k conversations² |
| --- | --- | --- | --- | --- | --- |
| Large API model (DeepSeek-V4.1-Flash) for everything | 100% | 0% | 0.8% | 3.8 s / 10.2 s | $1.07–2.14 |
| Base Qwen2.5-3B, local | 38.2% (34.0–42.8) | 100% | 4.8% | 2.4 s / 9.0 s | $0 |
| Fine-tuned 3B, local | 90.4% (87.6–93.0) | 100% | 1.8% | 3.7 s / 9.4 s | $0 |
| Fine-tuned 3B + gateway guards | 93.4% (91.2–95.6) | 100% | 1.2% | 3.9 s / 9.8 s | $0 |
| **Fine-tuned 3B + gateway + escalation** | **95.4% (93.4–97.2)** | **94%** | **0.4%** | **3.7 s / 10.6 s** | **$0.06–0.12** |

¹ An order or transfer nobody asked for or with the wrong item or quantity, or a number in the
answer that no tool returned. Orders are only ever drafts, so none of them executes without the
manager. ² DeepSeek list prices, Oct 2026 ($0.15–0.30 / M input, $0.60–1.20 / M output) and measured
tokens per conversation.

By scenario type, the full system scores **98%** on everyday questions, **98%** on the hard split
(typos, two products, follow-ups), **100%** on off-topic requests (declined without a tool call),
**100%** on vague and "should I order?" requests, and **82%** on the two tools held out of training.
It never placed an order nobody asked for.

### 1.5B vs 3B: same data, same recipe

| | Qwen2.5-1.5B | Qwen2.5-3B |
| --- | --- | --- |
| Base model, untrained | 28.8% | 38.2% |
| Fine-tuned, alone | 67.4% (8-bit) | **90.4%** (4-bit) |
| + gateway guards | 69.0% | **93.4%** |
| + gateway + escalation | 91.4% | **95.4%** |
| Answered locally (full system) | 75% | **94%** |
| Loss from 4-bit quantization | −8 points | −1.4 points |
| Served file / median latency | 1.65 GB / 3.2 s | 1.93 GB / 3.7 s |
| Training time (Kaggle T4) | 2.6 h | 4.7 h |

Doubling the model size, with no other change, took the fine-tuned model from 67% to 90%, made it
robust to 4-bit quantization, and cut the share of requests that need the large model from 25% to
6% for half a second more per conversation. The 3B is what `make up` serves.

<details>
<summary><b>Every model and precision tested</b></summary>

| Model | Task success | Everyday | Held-out tools | Off-topic | Vague / safety | Hard | Unrequested orders | Harmful | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash (API) | 100% | 100 | 100 | 100 | 100 | 100 | 0% | 0.8% | 3.8 s |
| Base 3B Q4_K_M | 38.2% | 12 | 48 | 87 | 52 | 21 | 0.2% | 4.8% | 2.4 s |
| **Fine-tuned 3B Q4_K_M** (served) | **90.4%** | 98 | 62 | 100 | 100 | 93 | 0.4% | 1.8% | 3.7 s |
| Fine-tuned 3B Q8_0 | 91.8% | 96 | 75 | 100 | 99 | 91 | 1.0% | 2.4% | 5.5 s |
| Base 1.5B F16 | 31.2% | 3 | 41 | 88 | 43 | 13 | 0% | 1.4% | 2.4 s |
| Base 1.5B Q4_K_M | 28.8% | 3 | 29 | 97 | 37 | 10 | 0% | 4.2% | 1.0 s |
| Fine-tuned 1.5B v1 F16 | 65.6% | 63 | 32 | 99 | 64 | 79 | 2.0% | 7.8% | 3.9 s |
| Fine-tuned 1.5B v1 Q4_K_M | 52.2% | 44 | 27 | 100 | 55 | 52 | 3.8% | 7.8% | 1.9 s |
| Fine-tuned 1.5B v2 F16 | 67.2% | 66 | 27 | 100 | 83 | 73 | 0.8% | 6.0% | 3.7 s |
| Fine-tuned 1.5B v2 Q8_0 | 67.4% | 67 | 25 | 100 | 84 | 74 | 0.8% | 7.6% | 2.4 s |
| Fine-tuned 1.5B v2 Q6_K | 62.8% | 59 | 22 | 100 | 76 | 71 | 1.0% | 8.0% | 2.0 s |
| Fine-tuned 1.5B v2 Q5_K_M | 64.0% | 62 | 25 | 100 | 81 | 66 | 0.8% | 6.6% | 2.0 s |
| Fine-tuned 1.5B v2 Q4_K_M | 59.4% | 51 | 21 | 100 | 76 | 67 | 0.2% | 5.4% | 1.8 s |

Summaries for every row: [`eval/results/`](eval/results/) (first-iteration runs in
[`schema_v1/`](eval/results/schema_v1/)).
</details>

## Safety, enforced in code

| Rule | How |
| --- | --- |
| 🔒 No order or transfer without the manager | Tools can only create **pending** drafts; only `POST /v1/orders/{id}/confirm` (the UI's Confirm button) places them. The model has no confirm tool. |
| 🔎 No invented products | A product ID must come from a catalog search in the same conversation, or the call is refused with "search first". |
| ✍️ Actions match the request | An order needs ordering language in the manager's message (or a "yes" to an offered order); "send / move … to another store" must be a transfer. Checked on 645 real orders: 1 false block. |
| 📦 Reasonable quantities | Orders above max(5 × high-case demand over lead time + review, 4 packs) are refused. |
| 🧯 Bad output can't break anything | One retry for malformed tool calls; at most 4 tool rounds and a time limit per request. |
| ↗️ Untrustworthy answers don't reach the manager | Unrecovered refusals or errors, limits or made-up numbers → the local attempt's drafts are cancelled and the request re-runs on the large model; a safe fallback reply if that fails too. |
| 👀 Read-only reads | Read tools use a read-only database connection. |

Plus API-key auth, per-client rate limiting, JSON logs with request IDs, `/health`, `/ready`, and
Prometheus metrics behind a 12-panel Grafana dashboard. 78 tests (15 drive the gateway with a
scripted fake model) run in CI on every push.

## What went wrong, and what fixed it

The first fine-tuned model (1.5B) scored 52% at 4-bit. Reading the failures and testing one
hypothesis at a time led to the changes that mattered most:

1. **The tool schema leaked an example ID.** The `item_id` description said "e.g. HOUSEHOLD_1_118",
   and the model put that exact ID into 70 tool calls instead of searching. Removing it cut invented
   IDs from 22% to 3%, but task success didn't move: the model still skipped the search. The schema
   fix removed a symptom, not the habit.
2. **The training mix taught the wrong habit.** General function-calling examples (39% of the first
   training set) show calling a function directly with arguments from the question. For a forecast
   question that meant calling `get_forecast` without looking the product up first, and once it
   drafted an order for the wrong product. The second version cut those examples to 500.
3. **"Should I order X?" drafted orders.** 300 extra training conversations that mention ordering but
   must not order brought unrequested orders from 3.8% to under 1%, and lifted the vague / safety
   scenarios from 55% to 84%.
4. **4-bit quantization erased half of the 1.5B's fine-tuning gain.** The base model loses nothing at
   4-bit, but the fine-tuned 1.5B lost 14 points; a precision sweep (4/5/6/8/16-bit) showed 8-bit
   matched full precision. The 3B, trained the same way, loses only 1.4 points at 4-bit.
5. **The gateway turns the remaining failures into safe ones.** The "search first" rule lets the model
   correct invented IDs (held-out tools: 62% → 77% for the 3B), and a write-intent rule (added after
   finding transfers turned into purchase orders and stock checks that ordered) cut wrong orders
   from 2.6% to 0.4% for the 1.5B.
6. **Then size: the same data on a 3B model** went from 67% to 90% on its own, with the base 3B at
   38%, so most of that is the fine-tuning on a bigger model, not a bigger model alone.

<p align="center"><img src="docs/loss_curve_3b.png" width="440" alt="Training and validation loss of the 3B run"> <img src="docs/loss_curve_v2.png" width="440" alt="Training and validation loss of the 1.5B run"></p>

## Performance

Measured on an Apple M5 MacBook Pro (10-core GPU, 24 GB), with the served 3B (Q4_K_M) behind the
gateway, escalation off (the laptop's own capacity). Real test-set questions, concurrent users,
2 minutes per level, zero errors:

| Setup | Users | p50 | p95 | Conversations / s |
| --- | --- | --- | --- | --- |
| **Metal, 4 slots** | **1** | **3.1 s** | **7.8 s** | 0.30 |
| Metal, 4 slots | 4 | 5.9 s | 17.9 s | 0.53 |
| **Metal, 4 slots** | **8** | 11.4 s | 21.9 s | **0.66** |
| Metal, 1 slot | 4 | 13.5 s | 28.1 s | 0.28 |
| CPU only, 2 slots | 1 | 18.5 s | 50.8 s | 0.04 |

One MacBook serves a single manager in ~3 s and up to ~0.66 conversations per second (about 2,400
an hour) with 4 parallel slots, which double throughput over a single slot. The GPU is ~6× faster
than CPU-only: prompt processing runs at ~3,100 tokens/s on Metal vs ~100 on the CPU (measured on
the 1.5B), and every request starts with ~2,000 prompt tokens. Those shared tokens (system prompt
and tool schemas) stay in llama.cpp's prompt cache, which is a large part of why local latency stays
low. The 1.5B is ~1.6× faster (about 1 conversation/s at 4 slots) but far less accurate; all runs are
in [`eval/results/`](eval/results/) (`loadtest_*.json`).

---

## Run it on your machine

### What you need

- **A Mac with Apple Silicon** (M1 or newer, 16 GB+ RAM recommended). Linux with an NVIDIA GPU works
  too with a CUDA build of llama.cpp; CPU-only works but is ~5× slower.
- [**uv**](https://docs.astral.sh/uv/) (Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **llama.cpp**: `brew install llama.cpp`
- **Docker**: [OrbStack](https://orbstack.dev) (free for personal use) or Docker Desktop
- A free [**Kaggle**](https://www.kaggle.com) account, for the sales data

### 1. Get the code

```bash
git clone https://github.com/meanderinghuman/storeops-copilot.git
cd storeops-copilot
uv sync
```

### 2. Get the sales data

1. Open the [M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
   and accept its rules.
2. Download `calendar.csv`, `sales_train_evaluation.csv` and `sell_prices.csv` into `data/raw/`.

### 3. Build the store database

```bash
uv run python -m store.build_db
```

This takes a few seconds and creates `data/store.db` (the product names are already in the repo, so
no API key is needed).

### 4. Download the fine-tuned model

```bash
uvx --from huggingface_hub hf download sidpnyc/storeops-copilot-qwen2.5-3b \
    storeops-3b-q4_k_m.gguf --local-dir models
```

On a Mac with 8 GB of RAM, use the smaller 1.5B instead (and start it in step 6 with
`make up MODEL=models/storeops-v2-q8_0.gguf ALIAS=storeops-v2-q8_0`):

```bash
uvx --from huggingface_hub hf download sidpnyc/storeops-copilot-qwen2.5-1.5b-v2 \
    storeops-v2-q8_0.gguf --local-dir models
```

### 5. Configure (optional)

```bash
cp .env.example .env
```

Out of the box the copilot runs **local-only**: the few requests the model can't handle get a polite
"couldn't complete" instead of escalating. To enable escalation, set `ESCALATION_BASE_URL`,
`ESCALATION_API_KEY` and `ESCALATION_MODEL` in `.env` to any OpenAI-compatible endpoint (for example
DeepSeek's API or OpenRouter).

### 6. Start everything

```bash
make up
```

This starts `llama-server` on the GPU and the gateway, forecast service, Prometheus and Grafana in
Docker. Then open:

| | |
| --- | --- |
| 💬 Copilot | http://localhost:8000 |
| 📖 API docs | http://localhost:8000/docs |
| 📊 Grafana | http://localhost:3000 |

Try the example questions in the sidebar. `make down` stops everything; `make reset-db` clears
orders you've placed.

### Run the tests

```bash
uv run pytest
```

<details>
<summary><b>Rebuild everything from scratch (data, training, evaluation)</b></summary>

| Step | Command |
| --- | --- |
| Generate questions | `uv run python -m datagen.queries --model <query writer>` |
| Teacher answers with real tool execution | `uv run python -m datagen.generate --model <teacher>` |
| Build the training mix | `uv run --group tokenize python -m datagen.mix` |
| Render with the chat template | `uv run --group tokenize python -m train.render` |
| Fine-tune | upload `data/kaggle_upload/` to Kaggle, run [`train/kaggle_train_3b.ipynb`](train/kaggle_train_3b.ipynb) (or [`kaggle_train.ipynb`](train/kaggle_train.ipynb) for 1.5B) |
| Build the test sets | `uv run python -m eval.build_testset` and `uv run python -m eval.build_hard` |
| Score a model | `uv run python -m eval.run --model <name> --base-url http://localhost:8080/v1 --api-key none` |
| Score the full system | add `--gateway escalation` |
| Load test | `loadtest/run_all.sh` |

Steps 1–2 need an OpenAI-compatible endpoint for the large models (`TEACHER_*` in `.env`).
</details>

## Project structure

```
store/       store database builder, the 8 tools, forecast service, system prompt
datagen/     question generation, agent loop, filters, training mix
train/       chat-template rendering and the Kaggle QLoRA notebooks (1.5B, 3B)
eval/        test sets, scorer, evaluation runner, results
serve/       gateway (FastAPI), web UI, Prometheus and Grafana config
loadtest/    concurrent-user load test
tests/       78 tests: tools, forecast, data pipeline, scorer, gateway safety
```

## Data and licenses

- **Code:** MIT.
- **Models:** the 3B ([`sidpnyc/storeops-copilot-qwen2.5-3b`](https://huggingface.co/sidpnyc/storeops-copilot-qwen2.5-3b))
  inherits the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)
  (research and non-commercial use); the 1.5B ([`sidpnyc/storeops-copilot-qwen2.5-1.5b-v2`](https://huggingface.co/sidpnyc/storeops-copilot-qwen2.5-1.5b-v2))
  is Apache 2.0.
- **Data:** M5 Forecasting (Walmart) via Kaggle; the raw files are not redistributed here. Product
  names, inventory, promotions and store names are synthetic.
- **Large models** used for data generation and escalation (DeepSeek-V4.1-Flash, GLM-5.3) were
  accessed through Virginia Tech's ARC LLM API.
