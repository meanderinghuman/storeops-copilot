"""Build the training mix: verified store conversations + a general tool-calling sample from xLAM.

    uv run --group tokenize python -m datagen.mix

xLAM (Salesforce/xlam-function-calling-60k, gated on Hugging Face; download to data/raw/xlam/)
keeps the model good at tool calling in general, so it doesn't overfit to six store tools. Its
examples are single-turn (query -> tool calls), converted to the same OpenAI/Qwen tool format.
Examples longer than --max-tokens under the Qwen2.5 chat template are dropped.
Outputs data/gen/train.jsonl and data/gen/val.jsonl, each row {id, source, kind, messages, tools}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datagen.runner import read_jsonl

GENERIC_SYSTEM = "You are a helpful assistant with access to tools. Call them when they help."

_SCALARS = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
            "dict": "object", "any": "string"}  # fmt: skip


def xlam_type(t: str) -> dict:
    """xLAM's Python-style type strings ('List[int]', 'str, optional') -> JSON Schema."""
    t = t.replace(", optional", "").strip()
    inner = re.fullmatch(r"(?:List|list|Set|Tuple)\[(.+)\]", t)
    if inner:
        return {"type": "array", "items": xlam_type(inner.group(1).split(",")[0])}
    if t.startswith(("Dict", "dict")):
        return {"type": "object"}
    if t.startswith(("Union", "Optional")):
        return {"type": "string"}
    return {"type": _SCALARS.get(t.lower(), "string")}


def xlam_tool(tool: dict) -> dict:
    props, required = {}, []
    for name, p in (tool.get("parameters") or {}).items():
        schema = {**xlam_type(str(p.get("type", "str"))), "description": p.get("description", "")}
        if "default" in p and p["default"] not in ("", None):
            schema["default"] = p["default"]
        props[name] = schema
        if "optional" not in str(p.get("type", "")) and "default" not in p:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def xlam_example(row: dict) -> dict:
    calls = json.loads(row["answers"])
    return {
        "id": f"xlam{row['id']:05d}",
        "source": "xlam",
        "kind": "general_tools",
        "messages": [
            {"role": "system", "content": GENERIC_SYSTEM},
            {"role": "user", "content": row["query"]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            },
        ],
        "tools": [xlam_tool(t) for t in json.loads(row["tools"])],
    }


def sample_xlam(path: Path, n: int, rng: random.Random) -> list[dict]:
    """Seeded sample of compact examples: 1-4 tools, 1-3 calls, short query."""
    rows = json.loads(path.read_text())
    ok = [
        r
        for r in rows
        if len(json.loads(r["tools"])) <= 4
        and 1 <= len(json.loads(r["answers"])) <= 3
        and len(r["query"]) <= 400
    ]
    return [xlam_example(r) for r in rng.sample(ok, n)]


def token_lengths(rows: list[dict], model: str) -> list[int]:
    """Exact length of each example under the student's chat template (needs `--group tokenize`)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    return [
        len(
            tok(tok.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False))[
                "input_ids"
            ]
        )
        for r in rows
    ]


def split(rows: list[dict], val_frac: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Stratified by (source, kind) so validation covers every request kind."""
    train, val = [], []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["source"], r["kind"]), []).append(r)
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda r: r["id"])
        rng.shuffle(g)
        k = max(1, round(len(g) * val_frac))
        val += g[:k]
        train += g[k:]
    rng.shuffle(train)
    return train, val


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verified", type=Path, default=Path("data/gen/verified.jsonl"))
    p.add_argument(
        "--xlam", type=Path, default=Path("data/raw/xlam/xlam_function_calling_60k.json")
    )
    p.add_argument("--xlam-n", type=int, default=1500)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--out-dir", type=Path, default=Path("data/gen"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=4096, help="drop longer examples; 0 = off")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    a = p.parse_args(argv)

    rng = random.Random(a.seed)
    store = [{**r, "source": "store"} for r in read_jsonl(a.verified)]
    rows = store + sample_xlam(a.xlam, a.xlam_n, rng)
    if a.max_tokens:
        lengths = token_lengths(rows, a.tokenizer)
        dropped = [r["id"] for r, n in zip(rows, lengths) if n > a.max_tokens]
        rows = [r for r, n in zip(rows, lengths) if n <= a.max_tokens]
        print(f"dropped {len(dropped)} examples over {a.max_tokens} tokens: {dropped}")
    train, val = split(rows, a.val_frac, rng)
    for name, part in (("train", train), ("val", val)):
        with (a.out_dir / f"{name}.jsonl").open("w") as f:
            for r in part:
                f.write(
                    json.dumps({k: r[k] for k in ("id", "source", "kind", "messages", "tools")})
                    + "\n"
                )
        counts = Counter(f"{r['source']}/{r['kind']}" for r in part)
        print(f"{name}: {len(part)}  {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
