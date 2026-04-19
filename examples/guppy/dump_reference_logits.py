#!/usr/bin/env python3
"""Load Guppy PyTorch weights and export fixed-prompt reference logits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def format_prompt(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump fixed-prompt Guppy reference logits for stage A."
    )
    parser.add_argument(
        "--assets-dir",
        default="build/guppy/assets",
        help="Directory containing pytorch_model.bin/config.json/tokenizer.json",
    )
    parser.add_argument(
        "--guppylm-root",
        default=os.path.expanduser("~/guppylm"),
        help="Path to the local guppylm repository",
    )
    parser.add_argument(
        "--messages-json",
        default="examples/guppy/fixed_prompt_messages.json",
        help="JSON file containing a message list",
    )
    parser.add_argument(
        "--out-dir",
        default="build/guppy/reference",
        help="Directory for the JSON summary and .npy logits dump",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for reference execution (default: cpu)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="How many top logits to summarize in the JSON output",
    )
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).expanduser().resolve()
    guppy_root = Path(args.guppylm_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = assets_dir / "pytorch_model.bin"
    config_path = assets_dir / "config.json"
    tokenizer_path = assets_dir / "tokenizer.json"
    messages_path = Path(args.messages_json).expanduser().resolve()

    for required in (checkpoint_path, config_path, tokenizer_path, messages_path):
        if not required.is_file():
            raise FileNotFoundError(f"required file not found: {required}")

    sys.path.insert(0, str(guppy_root))
    try:
        from tokenizers import Tokenizer
        from guppylm.inference import GuppyInference
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "failed to import guppylm/tokenizers; ensure tokenizers is installed "
            "and --guppylm-root points to the local repository"
        ) from exc

    messages = json.loads(messages_path.read_text())
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages-json must contain a non-empty message list")

    prompt = format_prompt(messages)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    input_ids = tokenizer.encode(prompt).ids

    engine = GuppyInference(
        str(checkpoint_path),
        str(tokenizer_path),
        device=args.device,
    )
    model = engine.model
    device = torch.device(args.device)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(input_tensor)

    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    last_logits = logits_np[0, -1]
    top_k = min(args.top_k, int(last_logits.shape[0]))
    top_indices = np.argsort(last_logits)[-top_k:][::-1]

    npy_path = out_dir / "reference_last_token_logits.npy"
    np.save(npy_path, last_logits)

    summary = {
        "schema_version": 1,
        "messages_json": str(messages_path),
        "prompt": prompt,
        "input_length": len(input_ids),
        "input_ids": input_ids,
        "logits_shape": list(logits_np.shape),
        "last_token_logits_path": str(npy_path),
        "top_k": [
            {"token_id": int(idx), "logit": float(last_logits[idx])}
            for idx in top_indices
        ],
        "assets": {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "tokenizer": str(tokenizer_path),
        },
    }

    json_path = out_dir / "reference_logits.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"prompt tokens: {len(input_ids)}")
    print(f"logits shape: {tuple(logits_np.shape)}")
    print(f"wrote {json_path}")
    print(f"wrote {npy_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
