---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- gguf
- llama.cpp
- tool-calling
- function-calling
- agents
- qlora
- unsloth
datasets:
- Salesforce/xlam-function-calling-60k
language:
- en
pipeline_tag: text-generation
---

# StoreOps Copilot: Qwen2.5-1.5B fine-tuned for store-inventory tool calling (v2)

A 1.5B model that answers store managers' inventory questions by calling store tools
(`search_items`, `get_inventory`, `get_sales_history`, `get_forecast`, `get_reorder_suggestion`,
`create_purchase_order`). Built to run locally with llama.cpp behind a safety gateway that
escalates hard requests to a large model. Code, data pipeline, evaluation and gateway:
**[github.com/meanderinghuman/storeops-copilot](https://github.com/meanderinghuman/storeops-copilot)**.

## Files

| File | Use |
| --- | --- |
| `storeops-v2-q8_0.gguf` | **Recommended.** Same accuracy as F16 on our benchmark, half the size. |
| `Qwen2.5-1.5B-Instruct.F16.gguf` | Full precision. |
| `Qwen2.5-1.5B-Instruct.Q4_K_M.gguf` | Smallest; loses ~8 points of task success vs Q8_0. |

LoRA adapter: [`sidpnyc/storeops-copilot-qwen2.5-1.5b-v2-lora`](https://huggingface.co/sidpnyc/storeops-copilot-qwen2.5-1.5b-v2-lora).

```bash
llama-server -m storeops-v2-q8_0.gguf --jinja -ngl 99 -c 16384 -np 2   # --jinja is required for tool calls
```

Use Qwen2.5's chat template with tools (OpenAI-style `tools` in the request). The model was trained
with the system prompt in [`store/prompt.py`](https://github.com/meanderinghuman/storeops-copilot/blob/main/store/prompt.py)
and the tool schemas in [`store/tools.py`](https://github.com/meanderinghuman/storeops-copilot/blob/main/store/tools.py);
results below assume both.

## Results

500 scenarios at stores and with two tools never seen in training, temperature 0:

| | Task success (95% CI) |
| --- | --- |
| Base Qwen2.5-1.5B-Instruct (Q4_K_M) | 28.8% (24.8-33.0) |
| **This model (Q8_0)** | **67.4% (63.2-71.4)** |
| This model + the project's gateway (guards + escalation to a large model) | 91.4% (89.0-93.8), 75% answered locally |
| DeepSeek-V4.1-Flash (teacher) | 100% |

By split for this model alone: seen tools 67%, held-out tools 25%, out-of-scope 100% (declines
without calling tools), ambiguous / write-safety 84%, hard (typos, two products, follow-ups) 74%.
Unrequested orders 0.8%.

## Training

- **Data:** 2,645 execution-verified store conversations (questions by GLM-5.3, answered by
  DeepSeek-V4.1-Flash as an agent with every tool call executed against the store database, 94.6%
  kept by automatic filters), plus 500 xLAM function-calling examples. 2,986 train / 157 validation.
- **QLoRA** with Unsloth on a Kaggle T4: rank 16, alpha 16, all attention and MLP projections,
  lr 2e-4 linear, 5% warmup, batch 8, max sequence 4,096, 2 epochs (748 steps, 2.6 h), loss on
  assistant turns only. Validation loss 0.37 → 0.235.

## Limitations

- Trained for one schema of store tools; weak on tools it hasn't seen (25% task success alone).
- Fails about 1 in 3 requests on its own; meant to run behind a gateway that validates calls and
  escalates (see the GitHub repo). Never let it place orders without human confirmation.
- Product data is synthetic (real Walmart M5 sales with generated names); not evaluated on real users.
