---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
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

# StoreOps Copilot: Qwen2.5-3B fine-tuned for store-inventory tool calling

A 3B model that answers store managers' inventory questions by calling store tools
(`search_items`, `get_inventory`, `get_sales_history`, `get_forecast`, `get_reorder_suggestion`,
`create_purchase_order`). Built to run locally with llama.cpp behind a safety gateway that
escalates hard requests to a large model. Code, data pipeline, evaluation and gateway:
**[github.com/meanderinghuman/storeops-copilot](https://github.com/meanderinghuman/storeops-copilot)**.

## Files

| File | Use |
| --- | --- |
| `storeops-3b-q4_k_m.gguf` | **Recommended.** 90.4% alone, 1.9 GB, ~3.7 s per conversation on an M5 MacBook. |
| `Qwen2.5-3B-Instruct.Q8_0.gguf` | 91.8% alone, 3.3 GB, slower. |

LoRA adapter: [`sidpnyc/storeops-copilot-qwen2.5-3b-lora`](https://huggingface.co/sidpnyc/storeops-copilot-qwen2.5-3b-lora).
Smaller sibling: [`sidpnyc/storeops-copilot-qwen2.5-1.5b-v2`](https://huggingface.co/sidpnyc/storeops-copilot-qwen2.5-1.5b-v2).

```bash
llama-server -m storeops-3b-q4_k_m.gguf --jinja -ngl 99 -c 16384 -np 2   # --jinja is required for tool calls
```

Use Qwen2.5's chat template with tools (OpenAI-style `tools` in the request). The model was trained
with the system prompt in [`store/prompt.py`](https://github.com/meanderinghuman/storeops-copilot/blob/main/store/prompt.py)
and the tool schemas in [`store/tools.py`](https://github.com/meanderinghuman/storeops-copilot/blob/main/store/tools.py);
results below assume both.

## Results

500 scenarios at stores and with two tools never seen in training, temperature 0:

| | Task success (95% CI) |
| --- | --- |
| Base Qwen2.5-1.5B-Instruct (Q4_K_M) | 28.8% |
| Fine-tuned 1.5B (Q8_0), same data | 67.4% |
| **This model (Q4_K_M)** | **90.4% (87.6-93.0)** |
| This model + the project's gateway (guards, no large model) | 93.4% (91.2-95.6), all local |
| DeepSeek-V4.1-Flash (teacher) | 100% |

By split for this model alone: seen tools 98%, held-out tools 63%, out-of-scope 100%, ambiguous /
write-safety 100%, hard (typos, two products, follow-ups) 93%. Harmful actions 1.8%.

## Training

- **Data:** 2,645 execution-verified store conversations (questions by GLM-5.3, answered by
  DeepSeek-V4.1-Flash as an agent with every tool call executed against the store database, 94.6%
  kept by automatic filters), plus 500 xLAM function-calling examples. 2,986 train / 157 validation.
- **QLoRA** with Unsloth on a Kaggle T4: rank 16, alpha 16, all attention and MLP projections,
  lr 2e-4 linear, 5% warmup, batch 1 x 8, max sequence 4,096, 2 epochs (748 steps, 4.7 h), loss
  on assistant turns only. Validation loss 0.33 → 0.213.

## Limitations

- Trained for one schema of store tools; weakest on tools it hasn't seen (63% task success alone).
- Meant to run behind a gateway that validates calls (see the GitHub repo). Never let it place
  orders without human confirmation.
- Product data is synthetic (real Walmart M5 sales with generated names); not evaluated on real users.

## License

This model is a derivative of Qwen2.5-3B-Instruct and is released under the
[Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)
(research and non-commercial use). The 1.5B version is Apache 2.0.
