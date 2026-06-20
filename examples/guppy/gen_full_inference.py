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
    %progress_20 = arith.constant 20 : i32
    func.call @guppy_set_progress(%progress_20) : (i32) -> ()

{emit_layernorm("%x_in", "%ln1_gamma", "%ln1_beta", "%x_ln", "%ln_mean", "%ln_var", seq_len, d_model, tag="ln1")}
    %progress_21 = arith.constant 21 : i32
    func.call @guppy_set_progress(%progress_21) : (i32) -> ()

{emit_linear_with_bias("%x_ln", "%qkv_w", "%qkv_b", "%qkv", seq_len, three_dim, d_model, tag="qkv")}
    %progress_22 = arith.constant 22 : i32
    func.call @guppy_set_progress(%progress_22) : (i32) -> ()

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
    %progress_23 = arith.constant 23 : i32
    func.call @guppy_set_progress(%progress_23) : (i32) -> ()

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
    %progress_24 = arith.constant 24 : i32
    func.call @guppy_set_progress(%progress_24) : (i32) -> ()

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
    %progress_25 = arith.constant 25 : i32
    func.call @guppy_set_progress(%progress_25) : (i32) -> ()

    scf.for %h = %c0 to %c_heads step %c1 {{
      scf.for %i = %c0 to %c_seq step %c1 {{
        scf.for %d = %c0 to %c_head_dim step %c1 {{
          %flat = affine.apply affine_map<(d0, d1) -> (d0 * {head_dim} + d1)>(%h, %d)
          %value = memref.load %attn_heads[%h, %i, %d] : memref<{n_heads}x{seq_len}x{head_dim}xf32>
          memref.store %value, %attn_merge[%i, %flat] : memref<{seq_len}x{d_model}xf32>
        }}
      }}
    }}

    func.call @guppy_after_attn_merge() : () -> ()

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
    %progress_26 = arith.constant 26 : i32
    func.call @guppy_set_progress(%progress_26) : (i32) -> ()

{emit_layernorm("%x_out", "%ln2_gamma", "%ln2_beta", "%x_ln2", "%ln_mean", "%ln_var", seq_len, d_model, tag="ln2")}
    %progress_27 = arith.constant 27 : i32
    func.call @guppy_set_progress(%progress_27) : (i32) -> ()

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
    %progress_28 = arith.constant 28 : i32
    func.call @guppy_set_progress(%progress_28) : (i32) -> ()

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
    %progress_29 = arith.constant 29 : i32
    func.call @guppy_set_progress(%progress_29) : (i32) -> ()
    func.call @guppy_after_transformer_block() : () -> ()

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
    %progress_40 = arith.constant 40 : i32
    func.call @guppy_set_progress(%progress_40) : (i32) -> ()

{emit_layernorm("%input", "%gamma", "%beta", "%ln_out", "%ln_mean", "%ln_var", seq_len, d_model, tag="lm")}
    %progress_41 = arith.constant 41 : i32
    func.call @guppy_set_progress(%progress_41) : (i32) -> ()

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
    %progress_42 = arith.constant 42 : i32
    func.call @guppy_set_progress(%progress_42) : (i32) -> ()

    return
  }}"""


def gen_full_mlir(seq_len: int, d_model: int, ffn_hidden: int, vocab_size: int, n_heads: int) -> str:
    return f"""// Auto-generated Guppy full forward.
module {{
  func.func private @guppy_set_progress(i32)
  func.func private @guppy_after_transformer_block()
  func.func private @guppy_after_attn_merge()

{gen_embedding_mlir(seq_len, d_model, vocab_size)}

{gen_transformer_block_mlir(seq_len, d_model, ffn_hidden, n_heads)}

{gen_lm_head_mlir(seq_len, d_model, vocab_size)}
}}
"""


def gen_split_post_attn_mlir() -> str:
    return """// Auto-generated Guppy PCIe split post-attention helper.
module {
  func.func @guppy_split_noop() attributes {vortex.entry} {
    return
  }
}
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


def gen_weights_asm(
    blobs: list[tuple[str, Path]],
    *,
    seq_len: int,
    d_model: int,
    ffn_hidden: int,
    n_heads: int,
    input_length: int,
    default_expect_golden: int,
    default_split_stage: int,
) -> str:
    head_dim = d_model // n_heads
    hidden_bytes = seq_len * d_model * 4
    qkv_bytes = seq_len * (3 * d_model) * 4
    attn_tensor_bytes = n_heads * seq_len * head_dim * 4
    score_bytes = n_heads * seq_len * seq_len * 4
    ffn_hidden_bytes = seq_len * ffn_hidden * 4
    scalar_seq_bytes = seq_len * 4
    scalar_head_seq_bytes = n_heads * seq_len * 4
    parts = [
        '    .section .bss.guppy_runtime,"aw",@nobits',
        '    .balign 16',
        '    .global g_x_cur',
        'g_x_cur:',
        f'    .zero {hidden_bytes}',
        '    .global g_x_next',
        'g_x_next:',
        f'    .zero {hidden_bytes}',
        '    .global g_x_ln',
        'g_x_ln:',
        f'    .zero {hidden_bytes}',
        '    .global g_qkv',
        'g_qkv:',
        f'    .zero {qkv_bytes}',
        '    .global g_q',
        'g_q:',
        f'    .zero {attn_tensor_bytes}',
        '    .global g_k',
        'g_k:',
        f'    .zero {attn_tensor_bytes}',
        '    .global g_v',
        'g_v:',
        f'    .zero {attn_tensor_bytes}',
        '    .global g_score',
        'g_score:',
        f'    .zero {score_bytes}',
        '    .global g_prob',
        'g_prob:',
        f'    .zero {score_bytes}',
        '    .global g_attn_heads',
        'g_attn_heads:',
        f'    .zero {attn_tensor_bytes}',
        '    .global g_attn_merge',
        'g_attn_merge:',
        f'    .zero {hidden_bytes}',
        '    .global g_attn_out',
        'g_attn_out:',
        f'    .zero {hidden_bytes}',
        '    .global g_x_ln2',
        'g_x_ln2:',
        f'    .zero {hidden_bytes}',
        '    .global g_hidden',
        'g_hidden:',
        f'    .zero {ffn_hidden_bytes}',
        '    .global g_ln_mean',
        'g_ln_mean:',
        f'    .zero {scalar_seq_bytes}',
        '    .global g_ln_var',
        'g_ln_var:',
        f'    .zero {scalar_seq_bytes}',
        '    .global g_sm_max',
        'g_sm_max:',
        f'    .zero {scalar_head_seq_bytes}',
        '    .global g_sm_sum',
        'g_sm_sum:',
        f'    .zero {scalar_head_seq_bytes}',
        '',
        '    .section .data.guppy_runtime,"aw",@progbits',
        '    .balign 64',
        '    .global guppy_runtime_prompt_length',
        'guppy_runtime_prompt_length:',
        f'    .word {input_length}',
        '    .global guppy_runtime_expect_golden',
        'guppy_runtime_expect_golden:',
        f'    .word {default_expect_golden}',
        '    .global guppy_runtime_pcie_split_stage',
        'guppy_runtime_pcie_split_stage:',
        f'    .word {default_split_stage}',
        '    .global guppy_runtime_stage0_checkpoint',
        'guppy_runtime_stage0_checkpoint:',
        '    .word 0',
        '    .global guppy_progress_stage',
        'guppy_progress_stage:',
        '    .word 0',
        '    .global guppy_stage0_profile',
        'guppy_stage0_profile:',
        '    .zero 768',
        '    .zero 48',
        '',
        '    .text',
        '',
    ]
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
    attn_out_thread_mode: str,
    ffn_thread_mode: str,
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

    per_layer_helpers = []
    per_layer_calls = []
    for layer_idx in range(n_layers):
        prefix = f"blocks.{layer_idx}"
        per_layer_helpers.append(
            f"""\
static __attribute__((noinline)) void guppy_run_block_{layer_idx}(void) {{
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
}}"""
        )
        layer_copy = ""
        if layer_idx + 1 < n_layers:
            layer_copy = """\

  for (int i = 0; i < S * D; ++i)
    g_x_cur[i] = g_x_next[i];"""
        split_exit = f"""\

  if (guppy_is_control_lane() && guppy_runtime_pcie_split_stage == {layer_idx + 1}) {{
    __asm__ volatile("fence rw, rw" ::: "memory");
    guppy_fast_exit(0);
  }}"""
        per_layer_calls.append(
            f"""\
  guppy_runtime_current_layer = {layer_idx};
  guppy_run_block_{layer_idx}();
  guppy_runtime_current_layer = -1;
  guppy_control_progress({3 + layer_idx * 2});{split_exit}{layer_copy}"""
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
  guppy_output_last_token_argmax = best_idx;
  guppy_control_progress(6);"""

    final_hidden_expr = "g_x_next" if n_layers > 0 else "g_x_cur"
    final_copy_code = ""
    if n_layers > 0:
        final_copy_code = """\
  for (int i = 0; i < S * D; ++i)
    g_x_cur[i] = g_x_next[i];

"""

    chat_fast_code = f"""\
  if (!guppy_runtime_expect_golden) {{
    int prompt_len = guppy_runtime_prompt_length;
    if (prompt_len <= 0)
      prompt_len = 1;
    if (prompt_len > S)
      prompt_len = S;
    int last_token_index = prompt_len - 1;

    (void)guppy_lm_head_one(
        &{final_hidden_expr}[last_token_index * D],
        (float*){tensor_symbol('norm.weight')},
        (float*){tensor_symbol('norm.bias')},
        (float*){tensor_symbol('tok_emb.weight')});

    guppy_control_progress(6);
    guppy_fast_exit(0);
  }}

"""

    lm_head_one_code = """\
static __attribute__((always_inline)) inline float guppy_sanitize_logit(float value) {
  unsigned bits;
  __asm__ volatile("fmv.x.w %0, %1" : "=r"(bits) : "f"(value));
  unsigned mag = bits & 0x7fffffffu;
  if (mag > 0x44800000u)
    return -3.4028234663852886e38f;
  return value;
}

static __attribute__((noinline)) int guppy_lm_head_one(
    const float *input, const float *gamma, const float *beta,
    const float *tok_table) {
  guppy_set_progress(50);
  float mean = 0.0f;
  for (int j = 0; j < D; ++j)
    mean += input[j];
  mean *= 1.0f / (float)D;

  float var = 0.0f;
  for (int j = 0; j < D; ++j) {
    float diff = input[j] - mean;
    var += diff * diff;
  }
  var *= 1.0f / (float)D;

  float inv_std = 1.0f / sqrtf(var + 1.0e-5f);
  for (int j = 0; j < D; ++j)
    g_lm_one_ln_out[j] = (input[j] - mean) * inv_std * gamma[j] + beta[j];

  guppy_set_progress(51);
  int best_idx = 0;
  float best_val = 0.0f;
  for (int v = 0; v < V; ++v) {
    const float *row = tok_table + v * D;
    float sum = 0.0f;
    for (int j = 0; j < D; ++j)
      sum += g_lm_one_ln_out[j] * row[j];
    sum = guppy_sanitize_logit(sum);
    guppy_output_last_token_logits[v] = sum;
    if (v == 0 || sum > best_val) {
      best_val = sum;
      best_idx = v;
    }
  }
  guppy_output_last_token_argmax = best_idx;
  guppy_set_progress(52);
  return best_idx;
}
"""

    if attn_out_thread_mode == "warp4":
        warp4_attn_out_globals = """\
volatile int guppy_warp4_attn_out_num_threads = 1;
volatile int guppy_warp4_attn_out_expected_mask = 1;
volatile int guppy_warp4_attn_out_nonzero_mask = 1;
volatile int guppy_warp4_attn_out_status = 0;
volatile int guppy_warp4_attn_out_task_count[4] = {0, 0, 0, 0};
volatile int guppy_warp4_attn_out_row = 0;
const float * volatile guppy_warp4_attn_out_weight = 0;
const float * volatile guppy_warp4_attn_out_bias = 0;
"""
        warp4_attn_out_helper = """\
static __attribute__((always_inline)) inline void guppy_store_f32_bits(
    float *addr, float value) {
  unsigned bits;
  __asm__ volatile("fmv.x.w %0, %1" : "=r"(bits) : "f"(value));
  ((volatile unsigned*)addr)[0] = bits;
}

static void __attribute__((noinline)) guppy_attn_out_row_warp4_body(void) {
  int tid = (int)guppy_thread_id();
  int row = guppy_warp4_attn_out_row;
  const float *weight = guppy_warp4_attn_out_weight;
  const float *bias = guppy_warp4_attn_out_bias;
  int lanes = guppy_warp4_attn_out_num_threads;
  if (lanes <= 0)
    lanes = 1;

  for (int j = tid; j < D; j += lanes) {
    float sum = bias[j];
    for (int k = 0; k < D; ++k)
      sum += g_attn_merge[row * D + k] * weight[j * D + k];
    guppy_store_f32_bits(&g_attn_out[row * D + j], sum);
    guppy_store_f32_bits(&g_x_next[row * D + j], g_x_cur[row * D + j] + sum);
    guppy_warp4_attn_out_task_count[tid] += 1;
  }
}

static void guppy_attn_out_row_warp4(
    int row, const float *weight, const float *bias) {
  int lanes = vx_num_threads();
  if (lanes > 4)
    lanes = 4;
  if (lanes < 1)
    lanes = 1;

  guppy_warp4_attn_out_num_threads = lanes;
  guppy_warp4_attn_out_expected_mask = (1 << lanes) - 1;
  guppy_warp4_attn_out_nonzero_mask = 0;
  guppy_warp4_attn_out_status = 0;
  guppy_warp4_attn_out_task_count[0] = 0;
  guppy_warp4_attn_out_task_count[1] = 0;
  guppy_warp4_attn_out_task_count[2] = 0;
  guppy_warp4_attn_out_task_count[3] = 0;
  guppy_warp4_attn_out_row = row;
  guppy_warp4_attn_out_weight = weight;
  guppy_warp4_attn_out_bias = bias;

  vx_fence();
  vx_tmc(guppy_warp4_attn_out_expected_mask);
  guppy_attn_out_row_warp4_body();
  vx_fence();
  vx_tmc_one();

  int nonzero_mask = 0;
  for (int i = 0; i < lanes; ++i) {
    if (guppy_warp4_attn_out_task_count[i] != 0)
      nonzero_mask |= 1 << i;
  }
  guppy_warp4_attn_out_nonzero_mask = nonzero_mask;
  if (D >= lanes && nonzero_mask != guppy_warp4_attn_out_expected_mask)
    guppy_warp4_attn_out_status = 1;
  __asm__ volatile("fence rw, rw" ::: "memory");
}
"""
        attn_out_row_code = f"""\
  guppy_attn_out_row_warp4(
      row,
      (float*){tensor_symbol('blocks.0.attn.out.weight')},
      (float*){tensor_symbol('blocks.0.attn.out.bias')});"""
    else:
        warp4_attn_out_globals = ""
        warp4_attn_out_helper = ""
        attn_out_row_code = f"""\
  for (int j = 0; j < D; ++j) {{
    float sum = {tensor_symbol('blocks.0.attn.out.bias')}[j];
    for (int k = 0; k < D; ++k)
      sum += g_attn_merge[row * D + k] *
             {tensor_symbol('blocks.0.attn.out.weight')}[j * D + k];
    g_attn_out[row * D + j] = sum;
    g_x_next[row * D + j] = g_x_cur[row * D + j] + sum;
  }}"""

    if ffn_thread_mode == "warp4":
        warp4_ffn_globals = """\
volatile int guppy_warp4_ffn_num_threads = 1;
volatile int guppy_warp4_ffn_row = 0;
"""
        warp4_ffn_helper = f"""\
static __attribute__((always_inline)) inline void guppy_ffn_store_f32_bits(
    float *addr, float value) {{
  unsigned bits;
  __asm__ volatile("fmv.x.w %0, %1" : "=r"(bits) : "f"(value));
  ((volatile unsigned*)addr)[0] = bits;
}}

static __attribute__((always_inline)) inline void guppy_ffn_store_relu_f32_bits(
    float *addr, float value) {{
  unsigned bits;
  unsigned keep_mask;
  __asm__ volatile(
      "fmv.x.w %[bits], %[value]\\n\\t"
      "srai %[keep_mask], %[bits], 31\\n\\t"
      "xori %[keep_mask], %[keep_mask], -1\\n\\t"
      "and %[bits], %[bits], %[keep_mask]\\n\\t"
      "sw %[bits], 0(%[addr])\\n\\t"
      : [bits] "=&r"(bits), [keep_mask] "=&r"(keep_mask)
      : [value] "f"(value), [addr] "r"(addr)
      : "memory");
}}

static int guppy_warp4_ffn_lane_count(void) {{
  int lanes = vx_num_threads();
  if (lanes > 4)
    lanes = 4;
  if (lanes < 1)
    lanes = 1;
  return lanes;
}}

static void __attribute__((noinline)) guppy_ffn_up_row_warp4_body(void) {{
  int tid = (int)guppy_thread_id();
  int row = guppy_warp4_ffn_row;
  int lanes = guppy_warp4_ffn_num_threads;
  if (lanes <= 0)
    lanes = 1;

  for (int h = tid; h < FF; h += lanes) {{
    float sum = {tensor_symbol('blocks.0.ffn.up.bias')}[h];
    for (int j = 0; j < D; ++j)
      sum += g_x_ln2[row * D + j] *
             {tensor_symbol('blocks.0.ffn.up.weight')}[h * D + j];
    guppy_ffn_store_relu_f32_bits(&g_hidden[row * FF + h], sum);
  }}
}}

static void __attribute__((noinline)) guppy_ffn_down_row_warp4_body(void) {{
  int tid = (int)guppy_thread_id();
  int row = guppy_warp4_ffn_row;
  int lanes = guppy_warp4_ffn_num_threads;
  if (lanes <= 0)
    lanes = 1;

  for (int j = tid; j < D; j += lanes) {{
    float sum = {tensor_symbol('blocks.0.ffn.down.bias')}[j];
    for (int h = 0; h < FF; ++h)
      sum += g_hidden[row * FF + h] *
             {tensor_symbol('blocks.0.ffn.down.weight')}[j * FF + h];
    guppy_ffn_store_f32_bits(&g_x_next[row * D + j], g_x_next[row * D + j] + sum);
  }}
}}

static void guppy_ffn_up_row_warp4(int row) {{
  int lanes = guppy_warp4_ffn_lane_count();
  guppy_warp4_ffn_num_threads = lanes;
  guppy_warp4_ffn_row = row;
  vx_fence();
  vx_tmc((1 << lanes) - 1);
  guppy_ffn_up_row_warp4_body();
  vx_fence();
  vx_tmc_one();
  __asm__ volatile("fence rw, rw" ::: "memory");
}}

static void guppy_ffn_down_row_warp4(int row) {{
  int lanes = guppy_warp4_ffn_lane_count();
  guppy_warp4_ffn_num_threads = lanes;
  guppy_warp4_ffn_row = row;
  vx_fence();
  vx_tmc((1 << lanes) - 1);
  guppy_ffn_down_row_warp4_body();
  vx_fence();
  vx_tmc_one();
  __asm__ volatile("fence rw, rw" ::: "memory");
}}
"""
        ffn_up_row_code = "  guppy_ffn_up_row_warp4(row);"
        ffn_down_row_code = "  guppy_ffn_down_row_warp4(row);"
    else:
        warp4_ffn_globals = ""
        warp4_ffn_helper = ""
        ffn_up_row_code = f"""\
  for (int h = 0; h < FF; ++h) {{
    float sum = {tensor_symbol('blocks.0.ffn.up.bias')}[h];
    for (int j = 0; j < D; ++j)
      sum += g_x_ln2[row * D + j] *
             {tensor_symbol('blocks.0.ffn.up.weight')}[h * D + j];
    if (sum < 0.0f)
      sum = 0.0f;
    g_hidden[row * FF + h] = sum;
  }}"""
        ffn_down_row_code = f"""\
  for (int j = 0; j < D; ++j) {{
    float sum = {tensor_symbol('blocks.0.ffn.down.bias')}[j];
    for (int h = 0; h < FF; ++h)
      sum += g_hidden[row * FF + h] *
             {tensor_symbol('blocks.0.ffn.down.weight')}[j * FF + h];
    g_x_next[row * D + j] += sum;
  }}"""

    return f"""\
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include <vortex/Runtime/BoardXDMAABI.h>
#include <math.h>

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

#if defined(__clang__)
#define GUPPY_WRAPPER_O0 __attribute__((optnone))
#elif defined(__GNUC__)
#define GUPPY_WRAPPER_O0 __attribute__((optimize("O0")))
#else
#define GUPPY_WRAPPER_O0
#endif

static __attribute__((always_inline)) inline unsigned guppy_thread_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC0));
  return value;
}}

static __attribute__((always_inline)) inline unsigned guppy_warp_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC1));
  return value;
}}

static __attribute__((always_inline)) inline unsigned guppy_core_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC2));
  return value;
}}

static __attribute__((always_inline)) inline int guppy_is_primary_lane(void) {{
  return guppy_thread_id() == 0 && guppy_warp_id() == 0;
}}

static __attribute__((always_inline)) inline int guppy_is_control_lane(void) {{
  return guppy_is_primary_lane() && guppy_core_id() == 0;
}}

{os.linesep.join(weight_decls)}

extern float g_x_cur[S * D];
extern float g_x_next[S * D];
extern float g_x_ln[S * D];
extern float g_qkv[S * (3 * D)];
extern float g_q[H * S * HD];
extern float g_k[H * S * HD];
extern float g_v[H * S * HD];
extern float g_score[H * S * S];
extern float g_prob[H * S * S];
extern float g_attn_heads[H * S * HD];
extern float g_attn_merge[S * D];
extern float g_attn_out[S * D];
extern float g_x_ln2[S * D];
extern float g_hidden[S * FF];
extern float g_ln_mean[S];
extern float g_ln_var[S];
extern float g_sm_max[H * S];
extern float g_sm_sum[H * S];
extern volatile int guppy_runtime_prompt_length;
extern volatile int guppy_runtime_expect_golden;
extern volatile int guppy_runtime_pcie_split_stage;
extern volatile int guppy_runtime_stage0_checkpoint;
float guppy_output_logits[S * V];
float guppy_output_last_token_logits[V];
int guppy_output_last_token_argmax = -1;
extern volatile int guppy_progress_stage;
extern volatile unsigned guppy_stage0_profile[192];
static volatile int guppy_runtime_current_layer = -1;
static volatile int guppy_after_attn_merge_done = 0;
static float g_lm_one_ln_out[D];
static float g_lm_ln_out[S * D];
static float g_lm_ln_mean[S];
static float g_lm_ln_var[S];
{warp4_attn_out_globals}
{warp4_ffn_globals}

static __attribute__((always_inline)) inline void guppy_control_progress(int value) {{
  if (guppy_is_control_lane())
    guppy_progress_stage = value;
}}

static __attribute__((always_inline)) inline void guppy_host_visible_fence(void) {{
  vortex_board_xdma_host_visible_fence();
}}

static __attribute__((always_inline)) inline unsigned guppy_read_mcycle_lo(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xB00));
  return value;
}}

static __attribute__((always_inline)) inline unsigned guppy_read_mcycle_hi(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xB80));
  return value;
}}

static void guppy_stage0_profile_reset(void) {{
  if (!guppy_is_control_lane())
    return;
  guppy_stage0_profile[0] = 0x47505330u;
  guppy_stage0_profile[1] = 13;
  guppy_stage0_profile[2] = 4;
  guppy_stage0_profile[3] = 0;
  guppy_host_visible_fence();
}}

static __attribute__((always_inline)) inline unsigned guppy_stage0_profile_info(
    int lane) {{
  return ((unsigned)guppy_progress_stage & 0xffu) |
         (((unsigned)lane & 0xffu) << 8) |
         ((guppy_thread_id() & 0xffu) << 16) |
         ((guppy_warp_id() & 0x0fu) << 24) |
         ((guppy_core_id() & 0x0fu) << 28);
}}

static __attribute__((always_inline)) inline int guppy_stage0_profile_base(
    int slot) {{
  if (slot < 8)
    return 96 + slot * 4;
  if (slot == 8)
    return 84;
  return 128 + (slot - 9) * 4;
}}

static __attribute__((always_inline)) inline void guppy_stage0_profile_mark(
    int slot, unsigned code, int lane) {{
  if (slot < 0 || slot >= 15)
    return;
  const int base = guppy_stage0_profile_base(slot);
  const unsigned hi = guppy_read_mcycle_hi();
  const unsigned lo = guppy_read_mcycle_lo();
  guppy_stage0_profile[base + 0] = code;
  guppy_stage0_profile[base + 1] = lo;
  guppy_stage0_profile[base + 2] = hi;
  guppy_stage0_profile[base + 3] = guppy_stage0_profile_info(lane);
  if ((unsigned)(slot + 1) > guppy_stage0_profile[1])
    guppy_stage0_profile[1] = (unsigned)(slot + 1);
  guppy_host_visible_fence();
}}

static __attribute__((noreturn)) void guppy_fast_exit(int status) {{
  vortex_board_xdma_exit_if(status, guppy_is_control_lane());
}}

static __attribute__((noreturn)) void guppy_stage0_fast_exit(int status, int lane) {{
  vortex_board_xdma_exit_if(status, lane == 0);
}}

static __attribute__((always_inline)) inline void guppy_stage0_checkpoint(
    int code, int lane) {{
  if (guppy_runtime_stage0_checkpoint == code) {{
    if (lane == 0) {{
      guppy_progress_stage = code;
      guppy_host_visible_fence();
    }}
    guppy_stage0_fast_exit(0, lane);
  }}
}}

void guppy_set_progress(int value) {{
  guppy_control_progress(value);
  if (guppy_is_control_lane() &&
      !guppy_runtime_expect_golden && guppy_runtime_pcie_split_stage == value) {{
    guppy_host_visible_fence();
    guppy_fast_exit(0);
  }}
}}

static void guppy_split_stage0_attn_merge_kernel(void *arg) {{
  (void)arg;
  const int lane = threadIdx.x;
  const int lanes = blockDim.x;

  if (lane >= 0 && lane < 4)
    guppy_stage0_profile_mark(9 + lane, 100u + (unsigned)lane, lane);
  if (lane == 0) {{
    guppy_control_progress(1);
    guppy_stage0_profile_mark(0, 1, lane);
  }}
  guppy_stage0_checkpoint(1, lane);
  for (int task = lane; task < S * D; task += lanes) {{
    const int row = task / D;
    const int col = task - row * D;
    const int token = guppy_input_token_ids[row];
    g_x_cur[task] =
        guppy_tok_emb_weight[token * D + col] + guppy_pos_emb_weight[task];
  }}
  __syncthreads();
  if (lane == 0)
    guppy_stage0_profile_mark(1, 20, lane);

  if (lane == 0) {{
    guppy_control_progress(20);
  }}
  guppy_stage0_checkpoint(20, lane);
  for (int row = lane; row < S; row += lanes) {{
    float mean = 0.0f;
    for (int col = 0; col < D; ++col)
      mean += g_x_cur[row * D + col];
    mean *= 1.0f / (float)D;

    float var = 0.0f;
    for (int col = 0; col < D; ++col) {{
      float diff = g_x_cur[row * D + col] - mean;
      var += diff * diff;
    }}
    var *= 1.0f / (float)D;
    const float inv_std = 1.0f / sqrtf(var + 1.0e-5f);
    g_ln_mean[row] = mean;
    g_ln_var[row] = var;
    for (int col = 0; col < D; ++col) {{
      g_x_ln[row * D + col] =
          (g_x_cur[row * D + col] - mean) * inv_std *
              guppy_blocks_0_norm1_weight[col] +
          guppy_blocks_0_norm1_bias[col];
    }}
  }}
  __syncthreads();
  if (lane == 0)
    guppy_stage0_profile_mark(2, 21, lane);

  if (lane == 0) {{
    guppy_control_progress(21);
  }}
  guppy_stage0_checkpoint(21, lane);
  for (int task = lane; task < S * (3 * D); task += lanes) {{
    const int row = task / (3 * D);
    const int out_col = task - row * (3 * D);
    float sum = guppy_blocks_0_attn_qkv_bias[out_col];
    const float *weight_row = &guppy_blocks_0_attn_qkv_weight[out_col * D];
    const float *input_row = &g_x_ln[row * D];
    for (int k = 0; k < D; ++k)
      sum += input_row[k] * weight_row[k];
    g_qkv[task] = sum;
  }}
  __syncthreads();
  if (lane == 0)
    guppy_stage0_profile_mark(3, 22, lane);

  if (lane == 0) {{
    guppy_control_progress(22);
  }}
  guppy_stage0_checkpoint(22, lane);
  for (int task = lane; task < H * S * HD; task += lanes) {{
    const int d = task % HD;
    const int tmp = task / HD;
    const int row = tmp % S;
    const int head = tmp / S;
    const int flat = head * HD + d;
    g_q[task] = g_qkv[row * (3 * D) + flat];
    g_k[task] = g_qkv[row * (3 * D) + D + flat];
    g_v[task] = g_qkv[row * (3 * D) + 2 * D + flat];
  }}
  __syncthreads();
  if (lane == 0)
    guppy_stage0_profile_mark(4, 23, lane);

  if (lane == 0) {{
    guppy_control_progress(23);
  }}
  guppy_stage0_checkpoint(23, lane);
  const float scale = {1.0 / math.sqrt(head_dim):.17g}f;
  const unsigned neg_inf_bits = 0xff800000u;
  for (int head = 0; head < H; ++head) {{
    for (int row = 0; row < S; ++row) {{
      float mx = -__builtin_inff();
      for (int col = 0; col < S; ++col) {{
        float acc = 0.0f;
        for (int d = 0; d < HD; ++d) {{
          acc += g_q[(head * S + row) * HD + d] *
                 g_k[(head * S + col) * HD + d];
        }}
        float raw_score = acc * scale;
        unsigned raw_bits;
        __asm__ volatile("fmv.x.w %0, %1" : "=r"(raw_bits) : "f"(raw_score));
        const unsigned keep_mask = 0u - (unsigned)(col <= row);
        const unsigned score_bits =
            (raw_bits & keep_mask) | (neg_inf_bits & ~keep_mask);
        float score;
        __asm__ volatile("fmv.w.x %0, %1" : "=f"(score) : "r"(score_bits));
        g_score[(head * S + row) * S + col] = score;
        mx = fmaxf(mx, score);
      }}
      g_sm_max[head * S + row] = mx;

      float denom = 0.0f;
      for (int col = 0; col < S; ++col) {{
        float p = expf(g_score[(head * S + row) * S + col] - mx);
        g_prob[(head * S + row) * S + col] = p;
        denom += p;
      }}
      g_sm_sum[head * S + row] = denom;
      for (int col = 0; col < S; ++col)
        g_prob[(head * S + row) * S + col] /= denom;
    }}
  }}
  __syncthreads();
  if (lane == 0)
    guppy_stage0_profile_mark(5, 24, lane);

  if (lane == 0) {{
    guppy_control_progress(24);
  }}
  guppy_stage0_checkpoint(24, lane);
  for (int task = lane; task < H * S * HD; task += lanes) {{
    const int d = task % HD;
    const int tmp = task / HD;
    const int row = tmp % S;
    const int head = tmp / S;
    float acc = 0.0f;
    for (int col = 0; col < S; ++col) {{
      acc += g_prob[(head * S + row) * S + col] *
             g_v[(head * S + col) * HD + d];
    }}
    g_attn_heads[task] = acc;
  }}
  __syncthreads();
  if (lane == 0)
    guppy_stage0_profile_mark(6, 25, lane);

  if (lane == 0) {{
    guppy_control_progress(25);
  }}
  guppy_stage0_checkpoint(25, lane);
  for (int task = lane; task < H * S * HD; task += lanes) {{
    const int d = task % HD;
    const int tmp = task / HD;
    const int row = tmp % S;
    const int head = tmp / S;
    g_attn_merge[row * D + head * HD + d] = g_attn_heads[task];
  }}
  __syncthreads();
  if (lane == 0)
    guppy_stage0_profile_mark(7, 240, lane);
  guppy_stage0_checkpoint(240, lane);
}}

static __attribute__((noinline)) void guppy_run_split_stage0_attn_merge(void) {{
  guppy_stage0_profile_reset();
  uint32_t grid_dim = 1;
  uint32_t block_dim = 4;
  int rc = vx_spawn_threads(
      1, &grid_dim, &block_dim,
      (vx_kernel_func_cb)guppy_split_stage0_attn_merge_kernel, 0);
  if (rc != 0) {{
    guppy_control_progress(239);
    guppy_fast_exit(1);
  }}
  guppy_control_progress(240);
  guppy_stage0_profile_mark(8, 241, -1);
  guppy_host_visible_fence();
  guppy_fast_exit(0);
}}

{os.linesep.join(per_layer_helpers)}

{warp4_attn_out_helper}
{warp4_ffn_helper}

{lm_head_one_code}

void __attribute__((noinline)) guppy_after_attn_merge(void) {{
  if (guppy_runtime_expect_golden)
    return;
  if ({n_layers} != 1) {{
    if (guppy_is_control_lane() && guppy_runtime_pcie_split_stage == 1) {{
      guppy_control_progress(240);
      __asm__ volatile("fence rw, rw" ::: "memory");
      guppy_fast_exit(0);
    }}
    return;
  }}

  if (!guppy_is_primary_lane()) {{
    while (!guppy_after_attn_merge_done) {{
    }}
    return;
  }}
  if (!guppy_is_control_lane())
    return;

  if (guppy_runtime_pcie_split_stage == 1) {{
    guppy_control_progress(240);
    __asm__ volatile("fence rw, rw" ::: "memory");
    guppy_fast_exit(0);
  }}

  int prompt_len = guppy_runtime_prompt_length;
  if (prompt_len <= 0)
    prompt_len = 1;
  if (prompt_len > S)
    prompt_len = S;
  int row = prompt_len - 1;

  guppy_set_progress(251);
{attn_out_row_code}

  guppy_set_progress(252);
  float mean = 0.0f;
  for (int j = 0; j < D; ++j)
    mean += g_x_next[row * D + j];
  mean *= 1.0f / (float)D;

  float var = 0.0f;
  for (int j = 0; j < D; ++j) {{
    float diff = g_x_next[row * D + j] - mean;
    var += diff * diff;
  }}
  var *= 1.0f / (float)D;

  float inv_std = 1.0f / sqrtf(var + 1.0e-5f);
  for (int j = 0; j < D; ++j)
    g_x_ln2[row * D + j] =
        (g_x_next[row * D + j] - mean) * inv_std *
            {tensor_symbol('blocks.0.norm2.weight')}[j] +
        {tensor_symbol('blocks.0.norm2.bias')}[j];

  guppy_set_progress(253);
{ffn_up_row_code}

  guppy_set_progress(254);
{ffn_down_row_code}

  guppy_set_progress(255);
  (void)guppy_lm_head_one(
      &g_x_next[row * D],
      (float*){tensor_symbol('norm.weight')},
      (float*){tensor_symbol('norm.bias')},
      (float*){tensor_symbol('tok_emb.weight')});

  guppy_after_attn_merge_done = 1;
  __asm__ volatile("fence rw, rw" ::: "memory");
  guppy_control_progress(6);
  guppy_fast_exit(0);
}}

void guppy_after_transformer_block(void) {{
  if (guppy_runtime_expect_golden)
    return;
  if ({n_layers} != 1) {{
    if (guppy_is_control_lane() &&
        guppy_runtime_pcie_split_stage > 0 &&
        guppy_runtime_pcie_split_stage == guppy_runtime_current_layer + 1) {{
      __asm__ volatile("fence rw, rw" ::: "memory");
      guppy_fast_exit(0);
    }}
    return;
  }}

  int prompt_len = guppy_runtime_prompt_length;
  if (prompt_len <= 0)
    prompt_len = 1;
  if (prompt_len > S)
    prompt_len = S;
  int last_token_index = prompt_len - 1;

  (void)guppy_lm_head_one(
      &g_x_next[last_token_index * D],
      (float*){tensor_symbol('norm.weight')},
      (float*){tensor_symbol('norm.bias')},
      (float*){tensor_symbol('tok_emb.weight')});

  guppy_control_progress(6);
  guppy_fast_exit(0);
}}

int main() {{
  if (!guppy_runtime_expect_golden &&
      guppy_runtime_pcie_split_stage == 1) {{
    guppy_run_split_stage0_attn_merge();
  }}

  if (!guppy_is_primary_lane())
    return 0;
  if (!guppy_is_control_lane())
    return 0;

  if (!guppy_runtime_expect_golden && guppy_runtime_pcie_split_stage == 101) {{
    guppy_control_progress(101);
    __asm__ volatile("fence rw, rw" ::: "memory");
    guppy_fast_exit(0);
  }}

  guppy_control_progress(1);
  embedding(
      (int*){tensor_symbol('input.token_ids')},
      (float*){tensor_symbol('tok_emb.weight')},
      (float*){tensor_symbol('pos_emb.weight')},
      g_x_cur);

  guppy_control_progress(2);
  if (!guppy_runtime_expect_golden && guppy_runtime_pcie_split_stage == 2) {{
    __asm__ volatile("fence rw, rw" ::: "memory");
    guppy_fast_exit(0);
  }}
{os.linesep.join(per_layer_calls)}

{chat_fast_code}{final_copy_code}
  guppy_control_progress(4);
  lm_head(
      g_x_cur,
      (float*){tensor_symbol('norm.weight')},
      (float*){tensor_symbol('norm.bias')},
      (float*){tensor_symbol('tok_emb.weight')},
      guppy_output_logits,
      g_lm_ln_out, g_lm_ln_mean, g_lm_ln_var);

  guppy_control_progress(5);
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


def gen_split_post_attn_wrapper(
    seq_len: int,
    d_model: int,
    ffn_hidden: int,
    vocab_size: int,
    n_heads: int,
    input_length: int,
) -> str:
    del ffn_hidden, n_heads
    return f"""\
#include <vx_intrinsics.h>

#define S {seq_len}
#define D {d_model}
#define V {vocab_size}
#define PROMPT_LEN {input_length}

static __attribute__((always_inline)) inline unsigned guppy_thread_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC0));
  return value;
}}

static __attribute__((always_inline)) inline unsigned guppy_warp_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC1));
  return value;
}}

static __attribute__((always_inline)) inline unsigned guppy_core_id(void) {{
  unsigned value;
  __asm__ volatile("csrr %0, %1" : "=r"(value) : "i"(0xCC2));
  return value;
}}

static __attribute__((always_inline)) inline int guppy_is_primary_lane(void) {{
  return guppy_thread_id() == 0 && guppy_warp_id() == 0;
}}

static __attribute__((always_inline)) inline int guppy_is_control_lane(void) {{
  return guppy_is_primary_lane() && guppy_core_id() == 0;
}}

int guppy_split_input_token_ids[S];
float guppy_split_attn_merge_row[D];
static volatile int guppy_runtime_prompt_length = PROMPT_LEN;
static volatile int guppy_runtime_expect_golden = 0;
static volatile int guppy_progress_stage = 0;
int guppy_split_host_argmax = -1;
int guppy_output_last_token_argmax = -1;

static __attribute__((always_inline)) inline void guppy_control_progress(int value) {{
  if (guppy_is_control_lane())
    guppy_progress_stage = value;
}}

int main() {{
  if (!guppy_is_primary_lane())
    return 0;
  if (!guppy_is_control_lane())
    return 0;

  guppy_control_progress(250);
  guppy_output_last_token_argmax = guppy_split_host_argmax;

  guppy_control_progress(6);
  __asm__ volatile("fence rw, rw" ::: "memory");
  return 0;
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
    split_mlir = manifest.get("pcie_split_post_attn_mlir")
    split_wrapper = manifest.get("pcie_split_post_attn_wrapper")
    if split_mlir and split_wrapper:
        (out_dir / split_mlir).write_text(manifest.pop("_split_mlir_text"))
        (out_dir / split_wrapper).write_text(manifest.pop("_split_wrapper_text"))
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
    parser.add_argument(
        "--pcie-default-split-stage",
        type=int,
        default=None,
        help=(
            "默认写入 guppy_runtime_pcie_split_stage 的值。"
            "不指定时 layer_limit>1 使用 1，否则使用 0。"
        ),
    )
    parser.add_argument(
        "--attn-out-thread-mode",
        choices=("serial", "warp4"),
        default="serial",
        help="blocks.0.attn.out 单行 projection 的线程调度模式。",
    )
    parser.add_argument(
        "--ffn-thread-mode",
        choices=("serial", "warp4"),
        default="serial",
        help="blocks.0 FFN up/down 单行 projection 的线程调度模式。",
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
    default_split_stage = (
        int(args.pcie_default_split_stage)
        if args.pcie_default_split_stage is not None
        else (1 if layer_limit > 1 else 0)
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
        args.attn_out_thread_mode,
        args.ffn_thread_mode,
    )
    split_mlir_text = gen_split_post_attn_mlir()
    split_wrapper_text = gen_split_post_attn_wrapper(
        seq_len,
        int(cfg["d_model"]),
        int(cfg["ffn_hidden"]),
        int(cfg["vocab_size"]),
        int(cfg["n_heads"]),
        int(assets["input_length"]),
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

    asm_text = gen_weights_asm(
        blobs,
        seq_len=seq_len,
        d_model=int(cfg["d_model"]),
        ffn_hidden=int(cfg["ffn_hidden"]),
        n_heads=int(cfg["n_heads"]),
        input_length=int(assets["input_length"]),
        default_expect_golden=0 if layer_limit > 1 else 1,
        default_split_stage=default_split_stage,
    )

    manifest = {
        "schema_version": 1,
        "generator": "examples/guppy/gen_full_inference.py",
        "bundle_dir": str(bundle_dir),
        "sequence_length": seq_len,
        "layer_limit": layer_limit,
        "prompt_length": int(assets["input_length"]),
        "tolerance": args.tolerance,
        "attn_out_thread_mode": args.attn_out_thread_mode,
        "ffn_thread_mode": args.ffn_thread_mode,
        "pcie_default_split_stage": default_split_stage,
        "mlir": "full_inference.mlir",
        "wrapper": "full_inference_wrapper.c",
        "weights_asm": "full_inference_weights.S",
        "pcie_split_post_attn_mlir": "split_post_attn.mlir",
        "pcie_split_post_attn_wrapper": "split_post_attn_wrapper.c",
        "pcie_split_post_attn_out_dir": "out_split_post_attn",
        "blob_count": len(blobs),
        "golden_logits_shape": list(golden_logits.shape),
        "next_token_argmax": int(np.argmax(golden_logits[int(assets["input_length"]) - 1])),
        "_split_mlir_text": split_mlir_text,
        "_split_wrapper_text": split_wrapper_text,
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
