# Results under the first tool schema (iteration 1)

These runs used the original `item_id` description, which contained an example ID
("e.g. HOUSEHOLD_1_118"). The fine-tuned v1 model copied that example into tool calls, so the
schema was changed (see the commit "Drop the example ID from the item_id schema"). Every model was
then re-scored under the new schema; the current results are one level up in `eval/results/`.

| Model | Task success (95% CI) |
| --- | --- |
| DeepSeek-V4.1-Flash (API) | 99.8% (99.4-100) |
| Base Qwen2.5-1.5B F16 | 30.2% (26.0-34.6) |
| Base Qwen2.5-1.5B Q4_K_M | 29.6% (25.6-34.0) |
| Fine-tuned v1 F16 | 67.1% (63.1-71.3) |
| Fine-tuned v1 Q4_K_M | 50.3% (45.9-54.7) |
