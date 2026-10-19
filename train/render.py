"""Render the training mix to plain text with the student's own chat template.

    uv run --group tokenize python -m train.render

Done locally, not on Kaggle: the examples have very different tool schemas, and loading nested
JSON with `datasets` would merge them into one Arrow schema (missing keys become nulls that then
leak into the rendered prompt). Rendering here also lets us check token lengths and the
assistant-turn markers that loss masking relies on before spending GPU time.

Writes data/gen/train_text.jsonl and val_text.jsonl as {"id", "source", "kind", "text"}.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ASSISTANT = "<|im_start|>assistant\n"  # loss is computed only after these markers
USER = "<|im_start|>user\n"  # user turns and tool results (Qwen renders those as user turns)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/gen"))
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-tokens", type=int, default=4096)
    a = p.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    for split in ("train", "val"):
        rows = [
            json.loads(line) for line in (a.data_dir / f"{split}.jsonl").read_text().splitlines()
        ]
        out, lengths = [], []
        for r in rows:
            text = tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False)
            n_assistant = sum(m["role"] == "assistant" for m in r["messages"])
            if text.count(ASSISTANT) != n_assistant:
                raise SystemExit(
                    f"{r['id']}: {text.count(ASSISTANT)} assistant markers, expected {n_assistant}"
                )
            n = len(tok(text)["input_ids"])
            if n > a.max_tokens:
                raise SystemExit(f"{r['id']} is {n} tokens > {a.max_tokens}; rebuild the mix")
            lengths.append(n)
            out.append({"id": r["id"], "source": r["source"], "kind": r["kind"], "text": text})
        path = a.data_dir / f"{split}_text.jsonl"
        path.write_text("".join(json.dumps(o) + "\n" for o in out))
        lengths.sort()
        print(f"{split}: {len(out)} examples -> {path} | tokens median {statistics.median(lengths):.0f}, "
              f"p95 {lengths[int(0.95 * len(lengths))]}, max {lengths[-1]}, total {sum(lengths):,}")  # fmt: skip


if __name__ == "__main__":
    main()
