#!/usr/bin/env python3
"""阶段 C：从 Guppy bundle 生成 full forward 的 MLIR / wrapper / rodata。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def array_to_c_initializer(arr: np.ndarray, per_line: int = 8) -> str:
    flat = arr.ravel()
    lines = []
    if flat.dtype.kind in ("i", "u"):
        for i in range(0, len(flat), per_line):
            chunk = flat[i:i + per_line]
            vals = ", ".join(str(int(v)) for v in chunk)
            lines.append(f"  {vals},")
    else:
        casted = flat.astype(np.float32)
        for i in range(0, len(casted), per_line):
            chunk = casted[i:i + per_line]
            vals = ", ".join(f"{float(v):.8e}f" for v in chunk)
            lines.append(f"  {vals},")
    return "\n".join(lines)


class BundleLoader:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir
        self.weights_index = load_json(bundle_dir / "weights_index.json")
        self.prompt = load_json(bundle_dir / "prompt.json")
        self.model_config = load_json(bundle_dir / "model_config.json")
        self.tensor_entries = {
            entry["name"]: entry for entry in self.weights_index["tensors"]
        }

    def tensor_entry(self, name: str) -> dict:
        entry = self.tensor_entries.get(name)
        if entry is None:
            raise KeyError(f"bundle 中缺少 tensor: {name}")
        return entry

    def tensor_path(self, name: str) -> Path:
        entry = self.tensor_entry(name)
        canonical = entry.get("alias_of")
        if canonical:
            entry = self.tensor_entry(canonical)
        file_name = entry.get("file")
        if not file_name:
            raise ValueError(f"tensor 没有 file 字段: {name}")
        return self.bundle_dir / file_name

    def load_tensor(self, name: str) -> np.ndarray:
        return np.load(self.tensor_path(name))


def layernorm_ref(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1.0e-5
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma + beta


def linear_ref(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def softmax_ref(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mx = x.max(axis=axis, keepdims=True)
    e = np.exp(x - mx)
    return e / e.sum(axis=axis, keepdims=True)


def attention_ref(
    x: np.ndarray,
    qkv_w: np.ndarray,
    qkv_b: np.ndarray,
    out_w: np.ndarray,
    out_b: np.ndarray,
    n_heads: int,
) -> np.ndarray:
    seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    qkv = linear_ref(x, qkv_w, qkv_b)
    qkv = qkv.reshape(seq_len, 3, n_heads, head_dim).transpose(1, 2, 0, 3)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    score = np.einsum("hsd,htd->hst", q, k) / math.sqrt(head_dim)
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    score = np.where(mask[None, :, :], score, -np.inf)
    prob = softmax_ref(score, axis=-1)
    attn = np.einsum("hst,htd->hsd", prob, v)
    merged = attn.transpose(1, 0, 2).reshape(seq_len, d_model)
    return linear_ref(merged, out_w, out_b)


def ffn_ref(
    x: np.ndarray,
    up_w: np.ndarray,
    up_b: np.ndarray,
    down_w: np.ndarray,
    down_b: np.ndarray,
) -> np.ndarray:
    hidden = linear_ref(x, up_w, up_b)
    hidden = np.maximum(hidden, 0.0)
    return linear_ref(hidden, down_w, down_b)


def block_ref(x: np.ndarray, layer: dict, n_heads: int) -> np.ndarray:
    x = x + attention_ref(
        layernorm_ref(x, layer["norm1.weight"], layer["norm1.bias"]),
        layer["attn.qkv.weight"],
        layer["attn.qkv.bias"],
        layer["attn.out.weight"],
        layer["attn.out.bias"],
        n_heads,
    )
    x = x + ffn_ref(
        layernorm_ref(x, layer["norm2.weight"], layer["norm2.bias"]),
        layer["ffn.up.weight"],
        layer["ffn.up.bias"],
        layer["ffn.down.weight"],
        layer["ffn.down.bias"],
    )
    return x


def build_reference(
    bundle: BundleLoader, sequence_length: int, layer_limit: int
) -> tuple[np.ndarray, dict]:
    cfg = bundle.model_config["normalized_config"]
    input_ids = list(bundle.prompt["input_ids"])
    input_len = len(input_ids)
    if input_len > sequence_length:
        raise ValueError(
            f"prompt 长度 {input_len} 超过生成序列长度 {sequence_length}"
        )

    padded_input = np.full(sequence_length, cfg["pad_id"], dtype=np.int32)
    padded_input[:input_len] = np.array(input_ids, dtype=np.int32)

    tok_emb = bundle.load_tensor("tok_emb.weight").astype(np.float32)
    pos_emb_full = bundle.load_tensor("pos_emb.weight").astype(np.float32)
    pos_emb = pos_emb_full[:sequence_length].copy()

    x = tok_emb[padded_input] + pos_emb
    x = x.astype(np.float32)

    layers = []
    for layer_idx in range(layer_limit):
        prefix = f"blocks.{layer_idx}"
        layer = {
            "norm1.weight": bundle.load_tensor(f"{prefix}.norm1.weight").astype(np.float32),
            "norm1.bias": bundle.load_tensor(f"{prefix}.norm1.bias").astype(np.float32),
            "attn.qkv.weight": bundle.load_tensor(f"{prefix}.attn.qkv.weight").astype(np.float32),
            "attn.qkv.bias": bundle.load_tensor(f"{prefix}.attn.qkv.bias").astype(np.float32),
            "attn.out.weight": bundle.load_tensor(f"{prefix}.attn.out.weight").astype(np.float32),
            "attn.out.bias": bundle.load_tensor(f"{prefix}.attn.out.bias").astype(np.float32),
            "norm2.weight": bundle.load_tensor(f"{prefix}.norm2.weight").astype(np.float32),
            "norm2.bias": bundle.load_tensor(f"{prefix}.norm2.bias").astype(np.float32),
            "ffn.up.weight": bundle.load_tensor(f"{prefix}.ffn.up.weight").astype(np.float32),
            "ffn.up.bias": bundle.load_tensor(f"{prefix}.ffn.up.bias").astype(np.float32),
            "ffn.down.weight": bundle.load_tensor(f"{prefix}.ffn.down.weight").astype(np.float32),
            "ffn.down.bias": bundle.load_tensor(f"{prefix}.ffn.down.bias").astype(np.float32),
        }
        x = block_ref(x, layer, cfg["n_heads"]).astype(np.float32)
        layers.append(layer)

    norm_weight = bundle.load_tensor("norm.weight").astype(np.float32)
    norm_bias = bundle.load_tensor("norm.bias").astype(np.float32)
    logits = linear_ref(
        layernorm_ref(x, norm_weight, norm_bias),
        tok_emb,
        None,
    ).astype(np.float32)

    assets = {
        "input_ids": padded_input,
        "input_length": input_len,
        "tok_emb": tok_emb,
        "pos_emb": pos_emb,
        "layers": layers,
        "norm_weight": norm_weight,
        "norm_bias": norm_bias,
        "golden_logits": logits,
    }
    return logits, assets


def emit_layernorm(
    input_name: str,
    gamma_name: str,
    beta_name: str,
    output_name: str,
    mean_name: str,
    var_name: str,
    seq_len: int,
    dim: int,
    zero_name: str = "%zero",
    eps_name: str = "%eps",
    inv_n_name: str = "%inv_n",
    tag: str = "ln",
) -> str:
    return f"""\
    linalg.fill ins({zero_name} : f32) outs({mean_name} : memref<{seq_len}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins({input_name} : memref<{seq_len}x{dim}xf32>)
      outs({mean_name} : memref<{seq_len}xf32>) {{
    ^bb0(%x_{tag}_mean: f32, %acc_{tag}_mean: f32):
      %sum_{tag}_mean = arith.addf %x_{tag}_mean, %acc_{tag}_mean : f32
      linalg.yield %sum_{tag}_mean : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins({mean_name} : memref<{seq_len}xf32>)
      outs({mean_name} : memref<{seq_len}xf32>) {{
    ^bb0(%sum_{tag}_scale: f32, %dummy_{tag}_scale: f32):
      %mean_{tag}_scale = arith.mulf %sum_{tag}_scale, {inv_n_name} : f32
      linalg.yield %mean_{tag}_scale : f32
    }}

    linalg.fill ins({zero_name} : f32) outs({var_name} : memref<{seq_len}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins({input_name}, {mean_name} : memref<{seq_len}x{dim}xf32>, memref<{seq_len}xf32>)
      outs({var_name} : memref<{seq_len}xf32>) {{
    ^bb0(%x_{tag}_var: f32, %mean_{tag}_var: f32, %acc_{tag}_var: f32):
      %diff_{tag}_var = arith.subf %x_{tag}_var, %mean_{tag}_var : f32
      %sq_{tag}_var = arith.mulf %diff_{tag}_var, %diff_{tag}_var : f32
      %sum_{tag}_var = arith.addf %sq_{tag}_var, %acc_{tag}_var : f32
      linalg.yield %sum_{tag}_var : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins({var_name} : memref<{seq_len}xf32>)
      outs({var_name} : memref<{seq_len}xf32>) {{
    ^bb0(%sum_{tag}_var_scale: f32, %dummy_{tag}_var_scale: f32):
      %var_{tag}_var_scale = arith.mulf %sum_{tag}_var_scale, {inv_n_name} : f32
      linalg.yield %var_{tag}_var_scale : f32
    }}

    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins({input_name}, {mean_name}, {var_name}, {gamma_name}, {beta_name} :
          memref<{seq_len}x{dim}xf32>, memref<{seq_len}xf32>, memref<{seq_len}xf32>,
          memref<{dim}xf32>, memref<{dim}xf32>)
      outs({output_name} : memref<{seq_len}x{dim}xf32>) {{
    ^bb0(%x_{tag}_norm: f32, %mean_{tag}_norm: f32, %var_{tag}_norm: f32,
         %gamma_{tag}_norm: f32, %beta_{tag}_norm: f32, %dummy_{tag}_norm: f32):
      %diff_{tag}_norm = arith.subf %x_{tag}_norm, %mean_{tag}_norm : f32
      %var_eps_{tag}_norm = arith.addf %var_{tag}_norm, {eps_name} : f32
      %inv_std_{tag}_norm = math.rsqrt %var_eps_{tag}_norm : f32
      %normed_{tag}_norm = arith.mulf %diff_{tag}_norm, %inv_std_{tag}_norm : f32
      %scaled_{tag}_norm = arith.mulf %normed_{tag}_norm, %gamma_{tag}_norm : f32
      %result_{tag}_norm = arith.addf %scaled_{tag}_norm, %beta_{tag}_norm : f32
      linalg.yield %result_{tag}_norm : f32
    }}"""


def emit_linear_with_bias(
    input_name: str,
    weight_name: str,
    bias_name: str | None,
    output_name: str,
    m: int,
    n: int,
    k: int,
    zero_name: str = "%zero",
    tag: str = "linear",
) -> str:
    fill = f"    linalg.fill ins({zero_name} : f32) outs({output_name} : memref<{m}x{n}xf32>)"
    reduce = f"""\
    linalg.generic {{
      indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                       affine_map<(i, j, k) -> (j, k)>,
                       affine_map<(i, j, k) -> (i, j)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    }} ins({input_name}, {weight_name} : memref<{m}x{k}xf32>, memref<{n}x{k}xf32>)
      outs({output_name} : memref<{m}x{n}xf32>) {{
    ^bb0(%lhs_{tag}: f32, %rhs_{tag}: f32, %acc_{tag}: f32):
      %prod_{tag} = arith.mulf %lhs_{tag}, %rhs_{tag} : f32
      %sum_{tag} = arith.addf %prod_{tag}, %acc_{tag} : f32
      linalg.yield %sum_{tag} : f32
    }}"""
    if bias_name is None:
        return fill + "\n" + reduce
    bias = f"""\
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins({output_name}, {bias_name} : memref<{m}x{n}xf32>, memref<{n}xf32>)
      outs({output_name} : memref<{m}x{n}xf32>) {{
    ^bb0(%value_{tag}: f32, %bias_{tag}: f32, %dummy_{tag}: f32):
      %result_{tag} = arith.addf %value_{tag}, %bias_{tag} : f32
      linalg.yield %result_{tag} : f32
    }}"""
    return fill + "\n" + reduce + "\n\n" + bias


def gen_embedding_mlir(seq_len: int, d_model: int, vocab_size: int) -> str:
    return f"""\
  func.func @embedding(%token_ids: memref<{seq_len}xi32>,
                       %tok_table: memref<{vocab_size}x{d_model}xf32>,
                       %pos_table: memref<{seq_len}x{d_model}xf32>,
                       %output: memref<{seq_len}x{d_model}xf32>)
      attributes {{vortex.entry}} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_seq = arith.constant {seq_len} : index
    %c_dim = arith.constant {d_model} : index

    scf.for %i = %c0 to %c_seq step %c1 {{
      %tok_i32 = memref.load %token_ids[%i] : memref<{seq_len}xi32>
      %tok = arith.index_cast %tok_i32 : i32 to index
      scf.for %j = %c0 to %c_dim step %c1 {{
        %tok_val = memref.load %tok_table[%tok, %j] : memref<{vocab_size}x{d_model}xf32>
        %pos_val = memref.load %pos_table[%i, %j] : memref<{seq_len}x{d_model}xf32>
        %sum = arith.addf %tok_val, %pos_val : f32
        memref.store %sum, %output[%i, %j] : memref<{seq_len}x{d_model}xf32>
      }}
    }}
    return
  }}"""


def gen_transformer_block_mlir(
    seq_len: int,
    d_model: int,
    ffn_hidden: int,
    n_heads: int,
) -> str:
    head_dim = d_model // n_heads
    inv_n = 1.0 / d_model
    scale = 1.0 / math.sqrt(head_dim)
    three_dim = 3 * d_model
    return f"""\
  func.func @transformer_block(
      %x_in: memref<{seq_len}x{d_model}xf32>,
      %x_out: memref<{seq_len}x{d_model}xf32>,
      %ln1_gamma: memref<{d_model}xf32>,
      %ln1_beta: memref<{d_model}xf32>,
      %qkv_w: memref<{three_dim}x{d_model}xf32>,
      %qkv_b: memref<{three_dim}xf32>,
      %attn_out_w: memref<{d_model}x{d_model}xf32>,
      %attn_out_b: memref<{d_model}xf32>,
      %ln2_gamma: memref<{d_model}xf32>,
      %ln2_beta: memref<{d_model}xf32>,
      %ffn_up_w: memref<{ffn_hidden}x{d_model}xf32>,
      %ffn_up_b: memref<{ffn_hidden}xf32>,
      %ffn_down_w: memref<{d_model}x{ffn_hidden}xf32>,
      %ffn_down_b: memref<{d_model}xf32>,
      %x_ln: memref<{seq_len}x{d_model}xf32>,
      %qkv: memref<{seq_len}x{three_dim}xf32>,
      %q: memref<{n_heads}x{seq_len}x{head_dim}xf32>,
      %k: memref<{n_heads}x{seq_len}x{head_dim}xf32>,
      %v: memref<{n_heads}x{seq_len}x{head_dim}xf32>,
      %score: memref<{n_heads}x{seq_len}x{seq_len}xf32>,
      %prob: memref<{n_heads}x{seq_len}x{seq_len}xf32>,
      %attn_heads: memref<{n_heads}x{seq_len}x{head_dim}xf32>,
      %attn_merge: memref<{seq_len}x{d_model}xf32>,
      %attn_out: memref<{seq_len}x{d_model}xf32>,
      %x_ln2: memref<{seq_len}x{d_model}xf32>,
      %hidden: memref<{seq_len}x{ffn_hidden}xf32>,
      %ln_mean: memref<{seq_len}xf32>,
      %ln_var: memref<{seq_len}xf32>,
      %sm_max: memref<{n_heads}x{seq_len}xf32>,
      %sm_sum: memref<{n_heads}x{seq_len}xf32>)
      attributes {{vortex.entry}} {{
    %zero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %inv_n = arith.constant {inv_n:.17g} : f32
    %scale = arith.constant {scale:.17g} : f32
    %neg_inf = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_seq = arith.constant {seq_len} : index
    %c_dim = arith.constant {d_model} : index
    %c_heads = arith.constant {n_heads} : index
    %c_head_dim = arith.constant {head_dim} : index
    %c_d_model = arith.constant {d_model} : index
    %c_two_d_model = arith.constant {2 * d_model} : index

{emit_layernorm("%x_in", "%ln1_gamma", "%ln1_beta", "%x_ln", "%ln_mean", "%ln_var", seq_len, d_model, tag="ln1")}

{emit_linear_with_bias("%x_ln", "%qkv_w", "%qkv_b", "%qkv", seq_len, three_dim, d_model, tag="qkv")}

    scf.for %h = %c0 to %c_heads step %c1 {{
      scf.for %i = %c0 to %c_seq step %c1 {{
        scf.for %d = %c0 to %c_head_dim step %c1 {{
          %flat = affine.apply affine_map<(d0, d1) -> (d0 * {head_dim} + d1)>(%h, %d)
          %k_idx = arith.addi %c_d_model, %flat : index
          %v_idx = arith.addi %c_two_d_model, %flat : index
          %q_val = memref.load %qkv[%i, %flat] : memref<{seq_len}x{three_dim}xf32>
          %k_val = memref.load %qkv[%i, %k_idx] : memref<{seq_len}x{three_dim}xf32>
          %v_val = memref.load %qkv[%i, %v_idx] : memref<{seq_len}x{three_dim}xf32>
          memref.store %q_val, %q[%h, %i, %d] : memref<{n_heads}x{seq_len}x{head_dim}xf32>
          memref.store %k_val, %k[%h, %i, %d] : memref<{n_heads}x{seq_len}x{head_dim}xf32>
          memref.store %v_val, %v[%h, %i, %d] : memref<{n_heads}x{seq_len}x{head_dim}xf32>
        }}
      }}
    }}

    linalg.fill ins(%zero : f32) outs(%score : memref<{n_heads}x{seq_len}x{seq_len}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(h, i, j, k) -> (h, i, k)>,
                       affine_map<(h, i, j, k) -> (h, j, k)>,
                       affine_map<(h, i, j, k) -> (h, i, j)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }} ins(%q, %k : memref<{n_heads}x{seq_len}x{head_dim}xf32>, memref<{n_heads}x{seq_len}x{head_dim}xf32>)
      outs(%score : memref<{n_heads}x{seq_len}x{seq_len}xf32>) {{
    ^bb0(%qv_score: f32, %kv_score: f32, %acc_score: f32):
      %prod_score = arith.mulf %qv_score, %kv_score : f32
      %sum_score = arith.addf %prod_score, %acc_score : f32
      linalg.yield %sum_score : f32
    }}

    linalg.generic {{
      indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>,
                       affine_map<(h, i, j) -> (h, i, j)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    }} ins(%score : memref<{n_heads}x{seq_len}x{seq_len}xf32>)
      outs(%score : memref<{n_heads}x{seq_len}x{seq_len}xf32>) {{
    ^bb0(%score_value_mask: f32, %score_dummy_mask: f32):
      %row = linalg.index 1 : index
      %col = linalg.index 2 : index
      %masked = arith.cmpi ule, %col, %row : index
      %scaled_mask = arith.mulf %score_value_mask, %scale : f32
      %result_mask = arith.select %masked, %scaled_mask, %neg_inf : f32
      linalg.yield %result_mask : f32
    }}

    linalg.fill ins(%neg_inf : f32) outs(%sm_max : memref<{n_heads}x{seq_len}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>,
                       affine_map<(h, i, j) -> (h, i)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    }} ins(%score : memref<{n_heads}x{seq_len}x{seq_len}xf32>)
      outs(%sm_max : memref<{n_heads}x{seq_len}xf32>) {{
    ^bb0(%value_max: f32, %acc_max: f32):
      %mx_max = arith.maximumf %value_max, %acc_max : f32
      linalg.yield %mx_max : f32
    }}

    linalg.generic {{
      indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>,
                       affine_map<(h, i, j) -> (h, i)>,
                       affine_map<(h, i, j) -> (h, i, j)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    }} ins(%score, %sm_max : memref<{n_heads}x{seq_len}x{seq_len}xf32>, memref<{n_heads}x{seq_len}xf32>)
      outs(%prob : memref<{n_heads}x{seq_len}x{seq_len}xf32>) {{
    ^bb0(%score_exp_value: f32, %score_exp_max: f32, %score_exp_dummy: f32):
      %shifted_exp = arith.subf %score_exp_value, %score_exp_max : f32
      %exp_value = math.exp %shifted_exp : f32
      linalg.yield %exp_value : f32
    }}

    linalg.fill ins(%zero : f32) outs(%sm_sum : memref<{n_heads}x{seq_len}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>,
                       affine_map<(h, i, j) -> (h, i)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    }} ins(%prob : memref<{n_heads}x{seq_len}x{seq_len}xf32>)
      outs(%sm_sum : memref<{n_heads}x{seq_len}xf32>) {{
    ^bb0(%value_sum: f32, %acc_sum: f32):
      %sum_softmax = arith.addf %value_sum, %acc_sum : f32
      linalg.yield %sum_softmax : f32
    }}

    linalg.generic {{
      indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>,
                       affine_map<(h, i, j) -> (h, i)>,
                       affine_map<(h, i, j) -> (h, i, j)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    }} ins(%prob, %sm_sum : memref<{n_heads}x{seq_len}x{seq_len}xf32>, memref<{n_heads}x{seq_len}xf32>)
      outs(%prob : memref<{n_heads}x{seq_len}x{seq_len}xf32>) {{
    ^bb0(%value_norm: f32, %sum_norm: f32, %dummy_norm: f32):
      %norm_value = arith.divf %value_norm, %sum_norm : f32
      linalg.yield %norm_value : f32
    }}

    linalg.fill ins(%zero : f32) outs(%attn_heads : memref<{n_heads}x{seq_len}x{head_dim}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(h, i, d, j) -> (h, i, j)>,
                       affine_map<(h, i, d, j) -> (h, j, d)>,
                       affine_map<(h, i, d, j) -> (h, i, d)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    }} ins(%prob, %v : memref<{n_heads}x{seq_len}x{seq_len}xf32>, memref<{n_heads}x{seq_len}x{head_dim}xf32>)
      outs(%attn_heads : memref<{n_heads}x{seq_len}x{head_dim}xf32>) {{
    ^bb0(%prob_attn: f32, %vv_attn: f32, %acc_attn: f32):
      %prod_attn = arith.mulf %prob_attn, %vv_attn : f32
      %sum_attn = arith.addf %prod_attn, %acc_attn : f32
      linalg.yield %sum_attn : f32
    }}

    scf.for %h = %c0 to %c_heads step %c1 {{
      scf.for %i = %c0 to %c_seq step %c1 {{
        scf.for %d = %c0 to %c_head_dim step %c1 {{
          %flat = affine.apply affine_map<(d0, d1) -> (d0 * {head_dim} + d1)>(%h, %d)
          %value = memref.load %attn_heads[%h, %i, %d] : memref<{n_heads}x{seq_len}x{head_dim}xf32>
          memref.store %value, %attn_merge[%i, %flat] : memref<{seq_len}x{d_model}xf32>
        }}
      }}
    }}

{emit_linear_with_bias("%attn_merge", "%attn_out_w", "%attn_out_b", "%attn_out", seq_len, d_model, d_model, tag="attn_out")}

    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%x_in, %attn_out : memref<{seq_len}x{d_model}xf32>, memref<{seq_len}x{d_model}xf32>)
      outs(%x_out : memref<{seq_len}x{d_model}xf32>) {{
    ^bb0(%lhs_res1: f32, %rhs_res1: f32, %dummy_res1: f32):
      %sum_res1 = arith.addf %lhs_res1, %rhs_res1 : f32
      linalg.yield %sum_res1 : f32
    }}

{emit_layernorm("%x_out", "%ln2_gamma", "%ln2_beta", "%x_ln2", "%ln_mean", "%ln_var", seq_len, d_model, tag="ln2")}

{emit_linear_with_bias("%x_ln2", "%ffn_up_w", "%ffn_up_b", "%hidden", seq_len, ffn_hidden, d_model, tag="ffn_up")}

    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%hidden : memref<{seq_len}x{ffn_hidden}xf32>)
      outs(%hidden : memref<{seq_len}x{ffn_hidden}xf32>) {{
    ^bb0(%relu_input: f32, %relu_dummy: f32):
      %relu_value = arith.maximumf %relu_input, %zero : f32
      linalg.yield %relu_value : f32
    }}

{emit_linear_with_bias("%hidden", "%ffn_down_w", "%ffn_down_b", "%attn_out", seq_len, d_model, ffn_hidden, tag="ffn_down")}

    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%x_out, %attn_out : memref<{seq_len}x{d_model}xf32>, memref<{seq_len}x{d_model}xf32>)
      outs(%x_out : memref<{seq_len}x{d_model}xf32>) {{
    ^bb0(%lhs_res2: f32, %rhs_res2: f32, %dummy_res2: f32):
      %sum_res2 = arith.addf %lhs_res2, %rhs_res2 : f32
      linalg.yield %sum_res2 : f32
    }}

    return
  }}"""


def gen_lm_head_mlir(seq_len: int, d_model: int, vocab_size: int) -> str:
    inv_n = 1.0 / d_model
    return f"""\
  func.func @lm_head(%input: memref<{seq_len}x{d_model}xf32>,
                     %gamma: memref<{d_model}xf32>,
                     %beta: memref<{d_model}xf32>,
                     %tok_table: memref<{vocab_size}x{d_model}xf32>,
                     %logits: memref<{seq_len}x{vocab_size}xf32>,
                     %ln_out: memref<{seq_len}x{d_model}xf32>,
                     %ln_mean: memref<{seq_len}xf32>,
                     %ln_var: memref<{seq_len}xf32>)
      attributes {{vortex.entry}} {{
    %zero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %inv_n = arith.constant {inv_n:.17g} : f32

{emit_layernorm("%input", "%gamma", "%beta", "%ln_out", "%ln_mean", "%ln_var", seq_len, d_model, tag="lm")}

    linalg.fill ins(%zero : f32) outs(%logits : memref<{seq_len}x{vocab_size}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                       affine_map<(i, j, k) -> (j, k)>,
                       affine_map<(i, j, k) -> (i, j)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    }} ins(%ln_out, %tok_table : memref<{seq_len}x{d_model}xf32>, memref<{vocab_size}x{d_model}xf32>)
      outs(%logits : memref<{seq_len}x{vocab_size}xf32>) {{
    ^bb0(%lhs_lm: f32, %rhs_lm: f32, %acc_lm: f32):
      %prod_lm = arith.mulf %lhs_lm, %rhs_lm : f32
      %sum_lm = arith.addf %prod_lm, %acc_lm : f32
      linalg.yield %sum_lm : f32
    }}

    return
  }}"""


def gen_full_mlir(seq_len: int, d_model: int, ffn_hidden: int, vocab_size: int, n_heads: int) -> str:
    return f"""// Auto-generated Guppy full forward.
module {{
{gen_embedding_mlir(seq_len, d_model, vocab_size)}

{gen_transformer_block_mlir(seq_len, d_model, ffn_hidden, n_heads)}

{gen_lm_head_mlir(seq_len, d_model, vocab_size)}
}}
"""


def tensor_symbol(name: str) -> str:
    safe = name.replace(".", "_")
    return f"guppy_{safe}"


def emit_incbin(label: str, path: Path, align: int = 16) -> str:
    quoted = str(path).replace("\\", "/")
    return f"""\
    .section .rodata
    .balign {align}
    .global {label}
{label}:
    .incbin "{quoted}"
"""


def write_blob(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def gen_weights_asm(blobs: list[tuple[str, Path]]) -> str:
    parts = ['    .text', '']
    for label, path in blobs:
        parts.append(emit_incbin(label, path))
        parts.append("")
    return "\n".join(parts)


def gen_wrapper(
    seq_len: int,
    d_model: int,
    ffn_hidden: int,
    vocab_size: int,
    n_layers: int,
    n_heads: int,
    input_length: int,
    tolerance: float,
) -> str:
    head_dim = d_model // n_heads

    weight_decls = [
        f"extern const int {tensor_symbol('input.token_ids')}[{seq_len}];",
        f"extern const float {tensor_symbol('tok_emb.weight')}[{vocab_size * d_model}];",
        f"extern const float {tensor_symbol('pos_emb.weight')}[{seq_len * d_model}];",
        f"extern const float {tensor_symbol('norm.weight')}[{d_model}];",
        f"extern const float {tensor_symbol('norm.bias')}[{d_model}];",
        f"extern const float {tensor_symbol('golden.logits')}[{seq_len * vocab_size}];",
    ]
    for layer_idx in range(n_layers):
        prefix = f"blocks.{layer_idx}"
        for name, size in [
            ("norm1.weight", d_model),
            ("norm1.bias", d_model),
            ("attn.qkv.weight", 3 * d_model * d_model),
            ("attn.qkv.bias", 3 * d_model),
            ("attn.out.weight", d_model * d_model),
            ("attn.out.bias", d_model),
            ("norm2.weight", d_model),
            ("norm2.bias", d_model),
            ("ffn.up.weight", ffn_hidden * d_model),
            ("ffn.up.bias", ffn_hidden),
            ("ffn.down.weight", d_model * ffn_hidden),
            ("ffn.down.bias", d_model),
        ]:
            weight_decls.append(
                f"extern const float {tensor_symbol(f'{prefix}.{name}')}[{size}];"
            )

    per_layer_calls = []
    for layer_idx in range(n_layers):
        prefix = f"blocks.{layer_idx}"
        per_layer_calls.append(
            f"""\
  transformer_block(
      g_x_cur, g_x_next,
      (float*){tensor_symbol(f"{prefix}.norm1.weight")},
      (float*){tensor_symbol(f"{prefix}.norm1.bias")},
      (float*){tensor_symbol(f"{prefix}.attn.qkv.weight")},
      (float*){tensor_symbol(f"{prefix}.attn.qkv.bias")},
      (float*){tensor_symbol(f"{prefix}.attn.out.weight")},
      (float*){tensor_symbol(f"{prefix}.attn.out.bias")},
      (float*){tensor_symbol(f"{prefix}.norm2.weight")},
      (float*){tensor_symbol(f"{prefix}.norm2.bias")},
      (float*){tensor_symbol(f"{prefix}.ffn.up.weight")},
      (float*){tensor_symbol(f"{prefix}.ffn.up.bias")},
      (float*){tensor_symbol(f"{prefix}.ffn.down.weight")},
      (float*){tensor_symbol(f"{prefix}.ffn.down.bias")},
      g_x_ln, g_qkv, g_q, g_k, g_v, g_score, g_prob, g_attn_heads,
      g_attn_merge, g_attn_out, g_x_ln2, g_hidden,
      g_ln_mean, g_ln_var, g_sm_max, g_sm_sum);

  for (int i = 0; i < S * D; ++i)
    g_x_cur[i] = g_x_next[i];"""
        )

    topk_code = """\
  int prompt_len = guppy_runtime_prompt_length;
  if (prompt_len <= 0)
    prompt_len = 1;
  if (prompt_len > S)
    prompt_len = S;
  int last_token_index = prompt_len - 1;

  int best_idx = 0;
  float best_val = guppy_output_logits[last_token_index * V];
  guppy_output_last_token_logits[0] = best_val;
  for (int v = 1; v < V; ++v) {
    float cur = guppy_output_logits[last_token_index * V + v];
    guppy_output_last_token_logits[v] = cur;
    if (cur > best_val) {
      best_val = cur;
      best_idx = v;
    }
  }
  guppy_output_last_token_argmax = best_idx;"""

    return f"""\
#include <vx_intrinsics.h>
#include <vx_print.h>

extern void embedding(int *token_ids, float *tok_table, float *pos_table, float *output);
extern void transformer_block(
    float *x_in, float *x_out,
    float *ln1_gamma, float *ln1_beta,
    float *qkv_w, float *qkv_b,
    float *attn_out_w, float *attn_out_b,
    float *ln2_gamma, float *ln2_beta,
    float *ffn_up_w, float *ffn_up_b,
    float *ffn_down_w, float *ffn_down_b,
    float *x_ln, float *qkv, float *q, float *k, float *v,
    float *score, float *prob, float *attn_heads, float *attn_merge,
    float *attn_out, float *x_ln2, float *hidden,
    float *ln_mean, float *ln_var, float *sm_max, float *sm_sum);
extern void lm_head(
    float *input, float *gamma, float *beta, float *tok_table,
    float *logits, float *ln_out, float *ln_mean, float *ln_var);

#define S {seq_len}
#define D {d_model}
#define FF {ffn_hidden}
#define V {vocab_size}
#define H {n_heads}
#define HD {head_dim}
#define PROMPT_LEN {input_length}
#define TOLERANCE {tolerance:.8e}f

{os.linesep.join(weight_decls)}

static float g_x_cur[S * D];
static float g_x_next[S * D];
static float g_x_ln[S * D];
static float g_qkv[S * (3 * D)];
static float g_q[H * S * HD];
static float g_k[H * S * HD];
static float g_v[H * S * HD];
static float g_score[H * S * S];
static float g_prob[H * S * S];
static float g_attn_heads[H * S * HD];
static float g_attn_merge[S * D];
static float g_attn_out[S * D];
static float g_x_ln2[S * D];
static float g_hidden[S * FF];
static float g_ln_mean[S];
static float g_ln_var[S];
static float g_sm_max[H * S];
static float g_sm_sum[H * S];
volatile int guppy_runtime_prompt_length = PROMPT_LEN;
volatile int guppy_runtime_expect_golden = 1;
float guppy_output_logits[S * V];
float guppy_output_last_token_logits[V];
int guppy_output_last_token_argmax = -1;
static float g_lm_ln_out[S * D];
static float g_lm_ln_mean[S];
static float g_lm_ln_var[S];

int main() {{
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  embedding(
      (int*){tensor_symbol('input.token_ids')},
      (float*){tensor_symbol('tok_emb.weight')},
      (float*){tensor_symbol('pos_emb.weight')},
      g_x_cur);

{os.linesep.join(per_layer_calls)}

  lm_head(
      g_x_cur,
      (float*){tensor_symbol('norm.weight')},
      (float*){tensor_symbol('norm.bias')},
      (float*){tensor_symbol('tok_emb.weight')},
      guppy_output_logits,
      g_lm_ln_out, g_lm_ln_mean, g_lm_ln_var);

  int pass = 1;
  float max_diff = 0.0f;
  if (guppy_runtime_expect_golden) {{
    for (int i = 0; i < S * V; ++i) {{
      float diff = guppy_output_logits[i] - {tensor_symbol('golden.logits')}[i];
      if (diff < 0.0f)
        diff = -diff;
      if (diff > max_diff)
        max_diff = diff;
      if (diff > TOLERANCE)
        pass = 0;
    }}
  }}

{topk_code}

  if (!guppy_runtime_expect_golden) {{
    vx_printf("guppy_next_token prompt_len=%d next_token=%d\\n",
              prompt_len, best_idx);
    return 0;
  }}

  if (pass) {{
    vx_printf("guppy_full_inference PASSED prompt_len=%d next_token=%d max_diff=%d.%04d\\n",
              prompt_len, best_idx, (int)max_diff,
              (int)((max_diff - (int)max_diff) * 10000));
    return 0;
  }}

  vx_printf("guppy_full_inference FAILED prompt_len=%d next_token=%d max_diff=%d.%04d\\n",
            prompt_len, best_idx, (int)max_diff,
            (int)((max_diff - (int)max_diff) * 10000));
  return 1;
}}
"""


def write_outputs(
    out_dir: Path,
    mlir_text: str,
    wrapper_text: str,
    asm_text: str,
    manifest: dict,
) -> None:
    (out_dir / "full_inference.mlir").write_text(mlir_text)
    (out_dir / "full_inference_wrapper.c").write_text(wrapper_text)
    (out_dir / "full_inference_weights.S").write_text(asm_text)
    (out_dir / "full_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Guppy stage-C full forward artifacts."
    )
    parser.add_argument(
        "--bundle-dir",
        default="build/guppy/export",
        help="阶段 B 导出的 bundle 目录",
    )
    parser.add_argument(
        "--out-dir",
        default="build/guppy/full_inference",
        help="阶段 C 产物输出目录",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="生成的静态序列长度，默认使用 max_seq_len",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0e-2,
        help="wrapper 校验 golden logits 的容差",
    )
    parser.add_argument(
        "--layer-limit",
        type=int,
        default=None,
        help="仅生成前 N 层，默认使用全部层数",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    blobs_dir = out_dir / "blobs"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = BundleLoader(bundle_dir)
    cfg = bundle.model_config["normalized_config"]
    seq_len = args.sequence_length or int(cfg["max_seq_len"])
    layer_limit = args.layer_limit or int(cfg["n_layers"])
    if seq_len <= 0 or seq_len > int(cfg["max_seq_len"]):
        raise ValueError(
            f"sequence-length 必须在 1..{cfg['max_seq_len']} 之间，当前是 {seq_len}"
        )
    if layer_limit <= 0 or layer_limit > int(cfg["n_layers"]):
        raise ValueError(
            f"layer-limit 必须在 1..{cfg['n_layers']} 之间，当前是 {layer_limit}"
        )

    golden_logits, assets = build_reference(bundle, seq_len, layer_limit)

    mlir_text = gen_full_mlir(
        seq_len,
        int(cfg["d_model"]),
        int(cfg["ffn_hidden"]),
        int(cfg["vocab_size"]),
        int(cfg["n_heads"]),
    )
    wrapper_text = gen_wrapper(
        seq_len,
        int(cfg["d_model"]),
        int(cfg["ffn_hidden"]),
        int(cfg["vocab_size"]),
        layer_limit,
        int(cfg["n_heads"]),
        int(assets["input_length"]),
        args.tolerance,
    )

    blobs: list[tuple[str, Path]] = []

    def add_blob(label: str, arr: np.ndarray) -> None:
        blob_path = blobs_dir / f"{label}.bin"
        write_blob(blob_path, np.ascontiguousarray(arr))
        blobs.append((label, blob_path.resolve()))

    add_blob(tensor_symbol("input.token_ids"), assets["input_ids"].astype(np.int32))
    add_blob(tensor_symbol("tok_emb.weight"), assets["tok_emb"].astype(np.float32))
    add_blob(tensor_symbol("pos_emb.weight"), assets["pos_emb"].astype(np.float32))
    add_blob(tensor_symbol("norm.weight"), assets["norm_weight"].astype(np.float32))
    add_blob(tensor_symbol("norm.bias"), assets["norm_bias"].astype(np.float32))
    add_blob(tensor_symbol("golden.logits"), assets["golden_logits"].astype(np.float32))

    for layer_idx, layer in enumerate(assets["layers"]):
        prefix = f"blocks.{layer_idx}"
        for key, value in layer.items():
            add_blob(tensor_symbol(f"{prefix}.{key}"), value.astype(np.float32))

    asm_text = gen_weights_asm(blobs)

    manifest = {
        "schema_version": 1,
        "generator": "examples/guppy/gen_full_inference.py",
        "bundle_dir": str(bundle_dir),
        "sequence_length": seq_len,
        "layer_limit": layer_limit,
        "prompt_length": int(assets["input_length"]),
        "tolerance": args.tolerance,
        "mlir": "full_inference.mlir",
        "wrapper": "full_inference_wrapper.c",
        "weights_asm": "full_inference_weights.S",
        "blob_count": len(blobs),
        "golden_logits_shape": list(golden_logits.shape),
        "next_token_argmax": int(np.argmax(golden_logits[int(assets["input_length"]) - 1])),
    }

    write_outputs(out_dir, mlir_text, wrapper_text, asm_text, manifest)

    print(f"wrote {out_dir / 'full_inference.mlir'}")
    print(f"wrote {out_dir / 'full_inference_wrapper.c'}")
    print(f"wrote {out_dir / 'full_inference_weights.S'}")
    print(f"wrote {out_dir / 'full_inference_manifest.json'}")
    print(f"sequence length: {seq_len}")
    print(f"layer limit: {layer_limit}")
    print(f"prompt length: {assets['input_length']}")
    print(f"golden logits shape: {golden_logits.shape}")
    print(
        "golden next-token argmax:",
        int(np.argmax(golden_logits[int(assets["input_length"]) - 1])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
