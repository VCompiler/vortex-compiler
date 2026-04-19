#!/usr/bin/env python3
"""阶段 B：把真实 Guppy 资产整理成可复用的导出 bundle。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def normalize_config(raw_cfg: dict) -> dict:
    cfg = {
        "model_type": raw_cfg.get("model_type", "guppylm"),
        "architectures": raw_cfg.get("architectures", ["GuppyLM"]),
        "vocab_size": int(raw_cfg.get("vocab_size", 4096)),
        "max_seq_len": int(
            raw_cfg.get("max_position_embeddings", raw_cfg.get("max_seq_len", 128))
        ),
        "d_model": int(raw_cfg.get("hidden_size", raw_cfg.get("d_model", 384))),
        "n_layers": int(
            raw_cfg.get("num_hidden_layers", raw_cfg.get("n_layers", 6))
        ),
        "n_heads": int(
            raw_cfg.get("num_attention_heads", raw_cfg.get("n_heads", 6))
        ),
        "ffn_hidden": int(
            raw_cfg.get("intermediate_size", raw_cfg.get("ffn_hidden", 768))
        ),
        "dropout": float(
            raw_cfg.get("hidden_dropout_prob", raw_cfg.get("dropout", 0.1))
        ),
        "pad_id": int(raw_cfg.get("pad_token_id", raw_cfg.get("pad_id", 0))),
        "bos_id": int(raw_cfg.get("bos_token_id", raw_cfg.get("bos_id", 1))),
        "eos_id": int(raw_cfg.get("eos_token_id", raw_cfg.get("eos_id", 2))),
    }
    if cfg["d_model"] % cfg["n_heads"] != 0:
        raise ValueError(
            f"d_model={cfg['d_model']} 不能整除 n_heads={cfg['n_heads']}"
        )
    cfg["head_dim"] = cfg["d_model"] // cfg["n_heads"]
    return cfg


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"不支持的 checkpoint 格式: {type(state_dict)!r}")
    return state_dict


def build_expected_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    d_model = cfg["d_model"]
    ffn_hidden = cfg["ffn_hidden"]
    vocab_size = cfg["vocab_size"]
    max_seq_len = cfg["max_seq_len"]
    n_layers = cfg["n_layers"]

    expected = {
        "tok_emb.weight": (vocab_size, d_model),
        "pos_emb.weight": (max_seq_len, d_model),
        "norm.weight": (d_model,),
        "norm.bias": (d_model,),
        "lm_head.weight": (vocab_size, d_model),
    }
    for layer in range(n_layers):
        prefix = f"blocks.{layer}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (d_model,),
                f"{prefix}.norm1.bias": (d_model,),
                f"{prefix}.attn.qkv.weight": (3 * d_model, d_model),
                f"{prefix}.attn.qkv.bias": (3 * d_model,),
                f"{prefix}.attn.out.weight": (d_model, d_model),
                f"{prefix}.attn.out.bias": (d_model,),
                f"{prefix}.norm2.weight": (d_model,),
                f"{prefix}.norm2.bias": (d_model,),
                f"{prefix}.ffn.up.weight": (ffn_hidden, d_model),
                f"{prefix}.ffn.up.bias": (ffn_hidden,),
                f"{prefix}.ffn.down.weight": (d_model, ffn_hidden),
                f"{prefix}.ffn.down.bias": (d_model,),
            }
        )
    return expected


def validate_state_dict(state_dict: dict[str, torch.Tensor], cfg: dict) -> dict:
    expected = build_expected_shapes(cfg)
    missing = []
    mismatched = []

    for name, shape in expected.items():
        tensor = state_dict.get(name)
        if tensor is None:
            missing.append(name)
            continue
        if tuple(tensor.shape) != shape:
            mismatched.append(
                {
                    "name": name,
                    "expected": list(shape),
                    "actual": list(tensor.shape),
                }
            )

    unexpected = sorted(set(state_dict.keys()) - set(expected.keys()))
    if missing or mismatched:
        lines = []
        if missing:
            lines.append("missing: " + ", ".join(missing))
        if mismatched:
            lines.append("mismatched: " + json.dumps(mismatched, ensure_ascii=False))
        raise ValueError("checkpoint 与 Guppy 配置不一致: " + "; ".join(lines))

    return {
        "expected_tensor_count": len(expected),
        "checkpoint_tensor_count": len(state_dict),
        "unexpected_keys": unexpected,
    }


def tensor_rel_path(name: str) -> Path:
    parts = name.split(".")
    return Path("weights").joinpath(*parts).with_suffix(".npy")


def tensor_sort_key(name: str) -> tuple[int, str]:
    priority = {
        "tok_emb.weight": 0,
        "pos_emb.weight": 1,
        "lm_head.weight": 2,
        "norm.weight": 3,
        "norm.bias": 4,
    }
    return (priority.get(name, 10), name)


def tensor_semantics(name: str) -> dict:
    if name == "tok_emb.weight":
        return {"kind": "token_embedding", "layer": None, "component": "tok_emb"}
    if name == "pos_emb.weight":
        return {"kind": "position_embedding", "layer": None, "component": "pos_emb"}
    if name == "lm_head.weight":
        return {"kind": "lm_head_weight", "layer": None, "component": "lm_head"}
    if name.startswith("norm."):
        return {"kind": "final_layernorm", "layer": None, "component": "norm"}
    if not name.startswith("blocks."):
        return {"kind": "unknown", "layer": None, "component": None}

    parts = name.split(".")
    layer = int(parts[1])
    component = ".".join(parts[2:-1])
    param = parts[-1]
    kind_map = {
        "norm1": "layernorm_pre_attn",
        "attn.qkv": "attention_qkv",
        "attn.out": "attention_out",
        "norm2": "layernorm_pre_ffn",
        "ffn.up": "ffn_up",
        "ffn.down": "ffn_down",
    }
    return {
        "kind": kind_map.get(component, "unknown"),
        "layer": layer,
        "component": component,
        "param": param,
    }


def dump_weights(state_dict: dict[str, torch.Tensor], out_dir: Path) -> dict:
    weights_root = out_dir / "weights"
    if weights_root.exists():
        shutil.rmtree(weights_root)

    seen_storage: dict[int, str] = {}
    entries = []
    alias_groups: dict[str, list[str]] = {}
    unique_tensor_count = 0
    unique_bytes = 0

    for name in sorted(state_dict.keys(), key=tensor_sort_key):
        tensor = state_dict[name].detach().cpu().contiguous()
        array = tensor.numpy()
        storage_id = tensor.untyped_storage().data_ptr()
        entry = {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "numel": int(array.size),
            "size_bytes": int(array.nbytes),
            **tensor_semantics(name),
        }

        canonical = seen_storage.get(storage_id)
        if canonical is None:
            rel_path = tensor_rel_path(name)
            abs_path = out_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(abs_path, array)
            seen_storage[storage_id] = name
            unique_tensor_count += 1
            unique_bytes += int(array.nbytes)
            entry["file"] = rel_path.as_posix()
            alias_groups[name] = [name]
        else:
            entry["alias_of"] = canonical
            alias_groups.setdefault(canonical, [canonical]).append(name)

        entries.append(entry)

    alias_sets = []
    for canonical, names in sorted(alias_groups.items()):
        if len(names) > 1:
            alias_sets.append({"canonical": canonical, "aliases": sorted(names[1:])})

    return {
        "schema_version": 1,
        "tensor_count": len(entries),
        "unique_tensor_count": unique_tensor_count,
        "unique_size_bytes": unique_bytes,
        "alias_groups": alias_sets,
        "tensors": entries,
    }


def copy_bundle_inputs(
    config_path: Path, tokenizer_path: Path, out_dir: Path
) -> tuple[str, str]:
    copied_config = "source_config.json"
    copied_tokenizer = "tokenizer.json"
    shutil.copy2(config_path, out_dir / copied_config)
    shutil.copy2(tokenizer_path, out_dir / copied_tokenizer)
    return copied_config, copied_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="阶段 B：导出 Guppy bundle（权重/配置/prompt/reference）。"
    )
    parser.add_argument(
        "--assets-dir",
        default="build/guppy/assets",
        help="包含 pytorch_model.bin/config.json/tokenizer.json 的目录",
    )
    parser.add_argument(
        "--guppylm-root",
        default=os.path.expanduser("~/guppylm"),
        help="本地 guppylm 仓库路径",
    )
    parser.add_argument(
        "--messages-json",
        default="examples/guppy/fixed_prompt_messages.json",
        help="输入 messages JSON",
    )
    parser.add_argument(
        "--out-dir",
        default="build/guppy/export",
        help="bundle 输出目录",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="reference forward 的 torch device",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="reference JSON 里保留的 top-k 个 logits",
    )
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).expanduser().resolve()
    guppy_root = Path(args.guppylm_root).expanduser().resolve()
    messages_path = Path(args.messages_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = assets_dir / "pytorch_model.bin"
    config_path = assets_dir / "config.json"
    tokenizer_path = assets_dir / "tokenizer.json"

    for required in (checkpoint_path, config_path, tokenizer_path, messages_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少必需文件: {required}")

    sys.path.insert(0, str(guppy_root))
    try:
        from tokenizers import Tokenizer
        from guppylm.config import GuppyConfig
        from guppylm.model import GuppyLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "导入 guppylm/tokenizers 失败；请检查 --guppylm-root 和 Python 环境"
        ) from exc

    raw_config = json.loads(config_path.read_text())
    normalized_config = normalize_config(raw_config)
    state_dict = load_state_dict(checkpoint_path)
    validation_summary = validate_state_dict(state_dict, normalized_config)

    guppy_config = GuppyConfig(
        vocab_size=normalized_config["vocab_size"],
        max_seq_len=normalized_config["max_seq_len"],
        d_model=normalized_config["d_model"],
        n_layers=normalized_config["n_layers"],
        n_heads=normalized_config["n_heads"],
        ffn_hidden=normalized_config["ffn_hidden"],
        dropout=normalized_config["dropout"],
        pad_id=normalized_config["pad_id"],
        bos_id=normalized_config["bos_id"],
        eos_id=normalized_config["eos_id"],
    )

    device = torch.device(args.device)
    model = GuppyLM(guppy_config).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        raise ValueError(
            "model.load_state_dict 缺少参数: "
            + ", ".join(incompatible.missing_keys)
        )
    model.eval()

    messages = json.loads(messages_path.read_text())
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages-json 必须是非空 message 列表")

    prompt = format_prompt(messages)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    input_ids = tokenizer.encode(prompt).ids
    if len(input_ids) > normalized_config["max_seq_len"]:
        raise ValueError(
            f"prompt 长度 {len(input_ids)} 超过 max_seq_len={normalized_config['max_seq_len']}"
        )

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(input_tensor)

    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    last_logits = logits_np[0, -1]
    top_k = min(args.top_k, int(last_logits.shape[0]))
    top_indices = np.argsort(last_logits)[-top_k:][::-1]

    copied_config, copied_tokenizer = copy_bundle_inputs(
        config_path, tokenizer_path, out_dir
    )

    weights_index = dump_weights(state_dict, out_dir)
    weights_index_path = out_dir / "weights_index.json"
    weights_index_path.write_text(
        json.dumps(weights_index, indent=2, ensure_ascii=False) + "\n"
    )

    full_logits_rel = "reference_logits.npy"
    last_logits_rel = "reference_last_token_logits.npy"
    np.save(out_dir / full_logits_rel, logits_np)
    np.save(out_dir / last_logits_rel, last_logits)

    prompt_json = {
        "schema_version": 1,
        "messages": messages,
        "prompt": prompt,
        "input_ids": input_ids,
        "input_length": len(input_ids),
        "special_token_ids": {
            "pad_id": normalized_config["pad_id"],
            "bos_id": normalized_config["bos_id"],
            "eos_id": normalized_config["eos_id"],
        },
        "tokenizer_json": copied_tokenizer,
    }
    prompt_path = out_dir / "prompt.json"
    prompt_path.write_text(json.dumps(prompt_json, indent=2, ensure_ascii=False) + "\n")

    reference_json = {
        "schema_version": 1,
        "input_length": len(input_ids),
        "logits_shape": list(logits_np.shape),
        "full_logits_npy": full_logits_rel,
        "last_token_logits_npy": last_logits_rel,
        "top_k": [
            {"token_id": int(idx), "logit": float(last_logits[idx])}
            for idx in top_indices
        ],
    }
    reference_path = out_dir / "reference_logits.json"
    reference_path.write_text(
        json.dumps(reference_json, indent=2, ensure_ascii=False) + "\n"
    )

    model_config_json = {
        "schema_version": 1,
        "normalized_config": normalized_config,
        "raw_config": raw_config,
        "source_config_json": copied_config,
        "param_count": int(sum(param.numel() for param in model.parameters())),
        "architecture": {
            "attention": {
                "multi_head": True,
                "fused_qkv": True,
                "causal_mask": True,
                "head_dim": normalized_config["head_dim"],
                "has_bias": True,
            },
            "ffn": {
                "activation": "relu",
                "has_bias": True,
            },
            "embeddings": {
                "tied_lm_head": True,
            },
        },
    }
    model_config_path = out_dir / "model_config.json"
    model_config_path.write_text(
        json.dumps(model_config_json, indent=2, ensure_ascii=False) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "export_stage": "B",
        "generator": "examples/guppy/guppy_to_vortex.py",
        "bundle_complete": True,
        "mlir_generated": False,
        "wrapper_generated": False,
        "weights_header_generated": False,
        "prompt_json": prompt_path.name,
        "model_config_json": model_config_path.name,
        "reference_json": reference_path.name,
        "weights_index_json": weights_index_path.name,
        "tokenizer_json": copied_tokenizer,
        "source_config_json": copied_config,
        "validation": validation_summary,
        "stage_c_requirements": {
            "multi_head_attention": True,
            "fused_qkv_projection": True,
            "causal_mask": True,
            "masked_softmax": True,
            "linear_bias": True,
            "relu_ffn": True,
            "tied_embedding_lm_head": True,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"prompt tokens: {len(input_ids)}")
    print(f"logits shape: {tuple(logits_np.shape)}")
    print(f"unique tensors: {weights_index['unique_tensor_count']}")
    print(f"alias groups: {len(weights_index['alias_groups'])}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
