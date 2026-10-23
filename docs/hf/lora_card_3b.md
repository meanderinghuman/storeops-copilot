---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
tags:
- lora
- qlora
- unsloth
- tool-calling
language:
- en
---

# StoreOps Copilot LoRA adapter (Qwen2.5-3B)

LoRA adapter (rank 16, all attention and MLP projections) for Qwen2.5-3B-Instruct, trained to answer
store-inventory questions by calling store tools. GGUF builds:
[`sidpnyc/storeops-copilot-qwen2.5-3b`](https://huggingface.co/sidpnyc/storeops-copilot-qwen2.5-3b).
Project, data pipeline and evaluation: [github.com/meanderinghuman/storeops-copilot](https://github.com/meanderinghuman/storeops-copilot).

License: [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE),
inherited from the base model (research and non-commercial use).
