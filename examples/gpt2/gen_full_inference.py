#!/usr/bin/env python3
"""Generate end-to-end GPT-2 inference: MLIR module, C wrapper, and weights header.

Chains: embedding -> N transformer blocks -> lm_head (final LN + vocab projection).

Usage:
    python3 gen_full_inference.py --seq 32 --dim 64 --ff 256 --vocab 256 \
            --layers 4 --seed 42 --out-dir DIR

Produces in DIR/:
    full_inference.mlir          -- single MLIR module with all kernel functions
    full_inference_wrapper.c     -- C wrapper calling kernels sequentially
    full_inference_weights.h     -- static const arrays for all weights + golden
"""

import argparse
import math
import os
import struct
import sys

import numpy as np

# Try scipy first for erf; fallback to math.erf element-wise
try:
    from scipy.special import erf as scipy_erf

    def gelu(x: np.ndarray) -> np.ndarray:
        return x * 0.5 * (1.0 + scipy_erf(x / math.sqrt(2.0)))
except ImportError:
    def gelu(x: np.ndarray) -> np.ndarray:
        result = np.empty_like(x)
        for idx in np.ndindex(x.shape):
            v = float(x[idx])
            result[idx] = v * 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
        return result


# ---------------------------------------------------------------------------
# Numpy reference implementation
# ---------------------------------------------------------------------------

def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma + beta


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mx = x.max(axis=axis, keepdims=True)
    e = np.exp(x - mx)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x: np.ndarray, Wq, Wk, Wv, Wo) -> np.ndarray:
    D = x.shape[-1]
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    score = Q @ K.T / math.sqrt(D)
    prob = softmax(score, axis=-1)
    attn = prob @ V
    return attn @ Wo


def mlp(x: np.ndarray, W1, W2) -> np.ndarray:
    h = x @ W1
    h = gelu(h)
    return h @ W2


def transformer_block(x: np.ndarray, weights: dict) -> np.ndarray:
    x_ln = layernorm(x, weights['ln1_gamma'], weights['ln1_beta'])
    attn_out = attention(x_ln, weights['Wq'], weights['Wk'],
                         weights['Wv'], weights['Wo'])
    x = x + attn_out
    x_ln2 = layernorm(x, weights['ln2_gamma'], weights['ln2_beta'])
    ff_out = mlp(x_ln2, weights['W1'], weights['W2'])
    x = x + ff_out
    return x


def embedding_ref(token_ids: np.ndarray, tok_table: np.ndarray,
                  pos_table: np.ndarray) -> np.ndarray:
    """token_ids: [S] int, tok_table: [V, D], pos_table: [S, D] -> [S, D]"""
    return tok_table[token_ids] + pos_table


def lm_head_ref(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                w_proj: np.ndarray) -> np.ndarray:
    """x: [S, D], gamma/beta: [D], w_proj: [D, V] -> logits: [S, V]"""
    x_ln = layernorm(x, gamma, beta)
    return x_ln @ w_proj


# ---------------------------------------------------------------------------
# Weight generation
# ---------------------------------------------------------------------------

WEIGHT_ORDER = [
    'ln1_gamma', 'ln1_beta',
    'Wq', 'Wk', 'Wv', 'Wo',
    'ln2_gamma', 'ln2_beta',
    'W1', 'W2',
]


def generate_layer_weights(D: int, FF: int, rng: np.random.RandomState) -> dict:
    return {
        'ln1_gamma': np.ones(D, dtype=np.float32),
        'ln1_beta': np.zeros(D, dtype=np.float32),
        'Wq': (0.02 * rng.randn(D, D)).astype(np.float32),
        'Wk': (0.02 * rng.randn(D, D)).astype(np.float32),
        'Wv': (0.02 * rng.randn(D, D)).astype(np.float32),
        'Wo': (0.02 * rng.randn(D, D)).astype(np.float32),
        'ln2_gamma': np.ones(D, dtype=np.float32),
        'ln2_beta': np.zeros(D, dtype=np.float32),
        'W1': (0.02 * rng.randn(D, FF)).astype(np.float32),
        'W2': (0.02 * rng.randn(FF, D)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# C helpers
# ---------------------------------------------------------------------------

def array_to_c_initializer(arr: np.ndarray, per_line: int = 8) -> str:
    flat = arr.ravel().astype(np.float32)
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        vals = ", ".join(f"{v:.8e}f" for v in chunk)
        lines.append(f"  {vals},")
    return "\n".join(lines)


def int_array_to_c_initializer(arr: np.ndarray, per_line: int = 16) -> str:
    flat = arr.ravel()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        vals = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"  {vals},")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def gen_embedding_mlir(S: int, D: int, V: int) -> str:
    """Generate the @embedding function body (no module wrapper)."""
    return f"""\
  func.func @embedding(%token_ids: memref<{S}xi32>,
                       %tok_table: memref<{V}x{D}xf32>,
                       %pos_table: memref<{S}x{D}xf32>,
                       %output: memref<{S}x{D}xf32>)
      attributes {{vortex.entry}} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_seq = arith.constant {S} : index
    %c_dim = arith.constant {D} : index

    scf.for %i = %c0 to %c_seq step %c1 {{
      %tok_id_i32 = memref.load %token_ids[%i] : memref<{S}xi32>
      %tok_id = arith.index_cast %tok_id_i32 : i32 to index
      scf.for %j = %c0 to %c_dim step %c1 {{
        %tok_val = memref.load %tok_table[%tok_id, %j] : memref<{V}x{D}xf32>
        %pos_val = memref.load %pos_table[%i, %j] : memref<{S}x{D}xf32>
        %sum = arith.addf %tok_val, %pos_val : f32
        memref.store %sum, %output[%i, %j] : memref<{S}x{D}xf32>
      }}
    }}
    return
  }}"""


def gen_transformer_block_mlir(S: int, D: int, FF: int) -> str:
    """Generate the @transformer_block function body (no module wrapper).

    Matches gen_transformer.py gen_mlir() exactly, but without the outer module.
    """
    inv_n = 1.0 / D
    scale = 1.0 / math.sqrt(D)
    inv_n_str = f"{inv_n:.17g}"
    scale_str = f"{scale:.11f}"

    return f"""\
  func.func @transformer_block(
      // Input / output
      %x_in: memref<{S}x{D}xf32>,
      %x_out: memref<{S}x{D}xf32>,
      // LayerNorm 1 weights
      %ln1_gamma: memref<{D}xf32>,
      %ln1_beta: memref<{D}xf32>,
      // Attention weights
      %wq: memref<{D}x{D}xf32>,
      %wk: memref<{D}x{D}xf32>,
      %wv: memref<{D}x{D}xf32>,
      %wo: memref<{D}x{D}xf32>,
      // LayerNorm 2 weights
      %ln2_gamma: memref<{D}xf32>,
      %ln2_beta: memref<{D}xf32>,
      // MLP weights
      %w1: memref<{D}x{FF}xf32>,
      %w2: memref<{FF}x{D}xf32>,
      // Scratch buffers
      %x_ln: memref<{S}x{D}xf32>,
      %q: memref<{S}x{D}xf32>,
      %k: memref<{S}x{D}xf32>,
      %v: memref<{S}x{D}xf32>,
      %score: memref<{S}x{S}xf32>,
      %prob: memref<{S}x{S}xf32>,
      %attn: memref<{S}x{D}xf32>,
      %attn_out: memref<{S}x{D}xf32>,
      %x_ln2: memref<{S}x{D}xf32>,
      %hidden: memref<{S}x{FF}xf32>,
      %ln_mean: memref<{S}xf32>,
      %ln_var: memref<{S}xf32>,
      %sm_max: memref<{S}xf32>,
      %sm_sum: memref<{S}xf32>)
      attributes {{vortex.entry}} {{

    // ---- Constants ----
    %zero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %inv_n = arith.constant {inv_n_str} : f32       // 1/{D}  (d_model)
    %scale = arith.constant {scale_str} : f32 // 1/sqrt({D})
    %neg_inf = arith.constant 0xFF800000 : f32 // -inf
    %half = arith.constant 0.5 : f32
    %inv_sqrt2 = arith.constant 0.70710678118 : f32
    %one = arith.constant 1.0 : f32

    // ================================================================
    // Step 1: LayerNorm(x_in, ln1_gamma, ln1_beta) -> x_ln
    // ================================================================

    // 1a: mean = sum(x_in) / N
    linalg.fill ins(%zero : f32) outs(%ln_mean : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%x_in : memref<{S}x{D}xf32>) outs(%ln_mean : memref<{S}xf32>) {{
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins(%ln_mean : memref<{S}xf32>) outs(%ln_mean : memref<{S}xf32>) {{
    ^bb0(%s: f32, %dummy: f32):
      %m = arith.mulf %s, %inv_n : f32
      linalg.yield %m : f32
    }}

    // 1b: var = sum((x_in - mean)^2) / N
    linalg.fill ins(%zero : f32) outs(%ln_var : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%x_in, %ln_mean : memref<{S}x{D}xf32>, memref<{S}xf32>)
      outs(%ln_var : memref<{S}xf32>) {{
    ^bb0(%x: f32, %mean: f32, %acc: f32):
      %diff = arith.subf %x, %mean : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %acc : f32
      linalg.yield %s : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins(%ln_var : memref<{S}xf32>) outs(%ln_var : memref<{S}xf32>) {{
    ^bb0(%s: f32, %dummy: f32):
      %vr = arith.mulf %s, %inv_n : f32
      linalg.yield %vr : f32
    }}

    // 1c: x_ln = (x_in - mean) / sqrt(var + eps) * gamma + beta
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%x_in, %ln_mean, %ln_var, %ln1_gamma, %ln1_beta :
          memref<{S}x{D}xf32>, memref<{S}xf32>, memref<{S}xf32>,
          memref<{D}xf32>, memref<{D}xf32>)
      outs(%x_ln : memref<{S}x{D}xf32>) {{
    ^bb0(%x: f32, %mean: f32, %var: f32, %g: f32, %b: f32, %dummy: f32):
      %diff = arith.subf %x, %mean : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %normed = arith.mulf %diff, %inv_std : f32
      %scaled = arith.mulf %normed, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    }}

    // ================================================================
    // Step 2: Q = x_ln @ Wq, K = x_ln @ Wk, V = x_ln @ Wv
    // ================================================================

    // Q = x_ln @ Wq
    linalg.fill ins(%zero : f32) outs(%q : memref<{S}x{D}xf32>)
    linalg.matmul ins(%x_ln, %wq : memref<{S}x{D}xf32>, memref<{D}x{D}xf32>)
        outs(%q : memref<{S}x{D}xf32>)

    // K = x_ln @ Wk
    linalg.fill ins(%zero : f32) outs(%k : memref<{S}x{D}xf32>)
    linalg.matmul ins(%x_ln, %wk : memref<{S}x{D}xf32>, memref<{D}x{D}xf32>)
        outs(%k : memref<{S}x{D}xf32>)

    // V = x_ln @ Wv
    linalg.fill ins(%zero : f32) outs(%v : memref<{S}x{D}xf32>)
    linalg.matmul ins(%x_ln, %wv : memref<{S}x{D}xf32>, memref<{D}x{D}xf32>)
        outs(%v : memref<{S}x{D}xf32>)

    // ================================================================
    // Step 3: score = Q @ K^T / sqrt(d)
    // ================================================================

    // Q @ K^T via linalg.generic with transposed indexing for K
    linalg.fill ins(%zero : f32) outs(%score : memref<{S}x{S}xf32>)
    linalg.generic {{
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,   // Q[m, k]
        affine_map<(m, n, k) -> (n, k)>,   // K[n, k]  (= K^T[k, n])
        affine_map<(m, n, k) -> (m, n)>    // score[m, n]
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    }} ins(%q, %k : memref<{S}x{D}xf32>, memref<{S}x{D}xf32>)
      outs(%score : memref<{S}x{S}xf32>) {{
    ^bb0(%q_val: f32, %k_val: f32, %acc: f32):
      %prod = arith.mulf %q_val, %k_val : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
    }}

    // Scale by 1/sqrt(d)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%score : memref<{S}x{S}xf32>) outs(%score : memref<{S}x{S}xf32>) {{
    ^bb0(%s: f32, %dummy: f32):
      %scaled = arith.mulf %s, %scale : f32
      linalg.yield %scaled : f32
    }}

    // ================================================================
    // Step 4: prob = softmax(score) along axis=-1
    // ================================================================

    // 4a: reduce_max
    linalg.fill ins(%neg_inf : f32) outs(%sm_max : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%score : memref<{S}x{S}xf32>) outs(%sm_max : memref<{S}xf32>) {{
    ^bb0(%a: f32, %acc: f32):
      %mx = arith.maximumf %a, %acc : f32
      linalg.yield %mx : f32
    }}

    // 4b: exp(score - max) -> prob
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%score, %sm_max : memref<{S}x{S}xf32>, memref<{S}xf32>)
      outs(%prob : memref<{S}x{S}xf32>) {{
    ^bb0(%s: f32, %mx: f32, %dummy: f32):
      %shifted = arith.subf %s, %mx : f32
      %e = math.exp %shifted : f32
      linalg.yield %e : f32
    }}

    // 4c: reduce_sum
    linalg.fill ins(%zero : f32) outs(%sm_sum : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%prob : memref<{S}x{S}xf32>) outs(%sm_sum : memref<{S}xf32>) {{
    ^bb0(%a: f32, %acc: f32):
      %s = arith.addf %a, %acc : f32
      linalg.yield %s : f32
    }}

    // 4d: div by sum
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%prob, %sm_sum : memref<{S}x{S}xf32>, memref<{S}xf32>)
      outs(%prob : memref<{S}x{S}xf32>) {{
    ^bb0(%e: f32, %s: f32, %dummy: f32):
      %r = arith.divf %e, %s : f32
      linalg.yield %r : f32
    }}

    // ================================================================
    // Step 5: attn = prob @ V
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%attn : memref<{S}x{D}xf32>)
    linalg.matmul ins(%prob, %v : memref<{S}x{S}xf32>, memref<{S}x{D}xf32>)
        outs(%attn : memref<{S}x{D}xf32>)

    // ================================================================
    // Step 6: attn_out = attn @ Wo
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%attn_out : memref<{S}x{D}xf32>)
    linalg.matmul ins(%attn, %wo : memref<{S}x{D}xf32>, memref<{D}x{D}xf32>)
        outs(%attn_out : memref<{S}x{D}xf32>)

    // ================================================================
    // Step 7: x_in = x_in + attn_out  (residual add, in-place)
    // ================================================================

    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%x_in, %attn_out : memref<{S}x{D}xf32>, memref<{S}x{D}xf32>)
      outs(%x_in : memref<{S}x{D}xf32>) {{
    ^bb0(%a: f32, %b: f32, %dummy: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }}

    // ================================================================
    // Step 8: LayerNorm(x_in, ln2_gamma, ln2_beta) -> x_ln2
    // ================================================================

    // 8a: mean = sum(x_in) / N
    linalg.fill ins(%zero : f32) outs(%ln_mean : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%x_in : memref<{S}x{D}xf32>) outs(%ln_mean : memref<{S}xf32>) {{
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins(%ln_mean : memref<{S}xf32>) outs(%ln_mean : memref<{S}xf32>) {{
    ^bb0(%s: f32, %dummy: f32):
      %m = arith.mulf %s, %inv_n : f32
      linalg.yield %m : f32
    }}

    // 8b: var = sum((x_in - mean)^2) / N
    linalg.fill ins(%zero : f32) outs(%ln_var : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%x_in, %ln_mean : memref<{S}x{D}xf32>, memref<{S}xf32>)
      outs(%ln_var : memref<{S}xf32>) {{
    ^bb0(%x: f32, %mean: f32, %acc: f32):
      %diff = arith.subf %x, %mean : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %acc : f32
      linalg.yield %s : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins(%ln_var : memref<{S}xf32>) outs(%ln_var : memref<{S}xf32>) {{
    ^bb0(%s: f32, %dummy: f32):
      %vr = arith.mulf %s, %inv_n : f32
      linalg.yield %vr : f32
    }}

    // 8c: x_ln2 = (x_in - mean) / sqrt(var + eps) * gamma + beta
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%x_in, %ln_mean, %ln_var, %ln2_gamma, %ln2_beta :
          memref<{S}x{D}xf32>, memref<{S}xf32>, memref<{S}xf32>,
          memref<{D}xf32>, memref<{D}xf32>)
      outs(%x_ln2 : memref<{S}x{D}xf32>) {{
    ^bb0(%x: f32, %mean: f32, %var: f32, %g: f32, %b: f32, %dummy: f32):
      %diff = arith.subf %x, %mean : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %normed = arith.mulf %diff, %inv_std : f32
      %scaled = arith.mulf %normed, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    }}

    // ================================================================
    // Step 9: hidden = x_ln2 @ W1  (d_model -> d_ff)
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%hidden : memref<{S}x{FF}xf32>)
    linalg.matmul ins(%x_ln2, %w1 : memref<{S}x{D}xf32>, memref<{D}x{FF}xf32>)
        outs(%hidden : memref<{S}x{FF}xf32>)

    // ================================================================
    // Step 10: hidden = GeLU(hidden)
    // ================================================================

    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%hidden : memref<{S}x{FF}xf32>) outs(%hidden : memref<{S}x{FF}xf32>) {{
    ^bb0(%x: f32, %dummy: f32):
      %x_sc = arith.mulf %x, %inv_sqrt2 : f32
      %erf_val = math.erf %x_sc : f32
      %one_erf = arith.addf %one, %erf_val : f32
      %x_half = arith.mulf %x, %half : f32
      %result = arith.mulf %x_half, %one_erf : f32
      linalg.yield %result : f32
    }}

    // ================================================================
    // Step 11: attn_out = hidden @ W2  (d_ff -> d_model)
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%attn_out : memref<{S}x{D}xf32>)
    linalg.matmul ins(%hidden, %w2 : memref<{S}x{FF}xf32>, memref<{FF}x{D}xf32>)
        outs(%attn_out : memref<{S}x{D}xf32>)

    // ================================================================
    // Step 12: x_out = x_in + attn_out  (residual add)
    // ================================================================

    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%x_in, %attn_out : memref<{S}x{D}xf32>, memref<{S}x{D}xf32>)
      outs(%x_out : memref<{S}x{D}xf32>) {{
    ^bb0(%a: f32, %b: f32, %dummy: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }}

    return
  }}"""


def gen_lm_head_mlir(S: int, D: int, V: int) -> str:
    """Generate the @lm_head function body (no module wrapper)."""
    inv_n = 1.0 / D
    inv_n_str = f"{inv_n:.17g}"

    return f"""\
  func.func @lm_head(%input: memref<{S}x{D}xf32>,
                     %gamma: memref<{D}xf32>, %beta: memref<{D}xf32>,
                     %w_proj: memref<{D}x{V}xf32>,
                     %logits: memref<{S}x{V}xf32>,
                     %ln_out: memref<{S}x{D}xf32>,
                     %ln_mean: memref<{S}xf32>, %ln_var: memref<{S}xf32>)
      attributes {{vortex.entry}} {{

    %zero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %inv_n = arith.constant {inv_n_str} : f32  // 1/{D}

    // LayerNorm: mean
    linalg.fill ins(%zero : f32) outs(%ln_mean : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%input : memref<{S}x{D}xf32>) outs(%ln_mean : memref<{S}xf32>) {{
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins(%ln_mean : memref<{S}xf32>) outs(%ln_mean : memref<{S}xf32>) {{
    ^bb0(%s: f32, %d: f32):
      %m = arith.mulf %s, %inv_n : f32
      linalg.yield %m : f32
    }}

    // LayerNorm: variance
    linalg.fill ins(%zero : f32) outs(%ln_var : memref<{S}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%input, %ln_mean : memref<{S}x{D}xf32>, memref<{S}xf32>)
      outs(%ln_var : memref<{S}xf32>) {{
    ^bb0(%x: f32, %mean: f32, %acc: f32):
      %diff = arith.subf %x, %mean : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %acc : f32
      linalg.yield %s : f32
    }}
    linalg.generic {{
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    }} ins(%ln_var : memref<{S}xf32>) outs(%ln_var : memref<{S}xf32>) {{
    ^bb0(%s: f32, %d: f32):
      %v = arith.mulf %s, %inv_n : f32
      linalg.yield %v : f32
    }}

    // LayerNorm: normalize
    linalg.generic {{
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%input, %ln_mean, %ln_var, %gamma, %beta :
          memref<{S}x{D}xf32>, memref<{S}xf32>, memref<{S}xf32>,
          memref<{D}xf32>, memref<{D}xf32>)
      outs(%ln_out : memref<{S}x{D}xf32>) {{
    ^bb0(%x: f32, %mean: f32, %var: f32, %g: f32, %b: f32, %dummy: f32):
      %diff = arith.subf %x, %mean : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %normed = arith.mulf %diff, %inv_std : f32
      %scaled = arith.mulf %normed, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    }}

    // Vocab projection: logits = ln_out @ w_proj
    linalg.fill ins(%zero : f32) outs(%logits : memref<{S}x{V}xf32>)
    linalg.matmul ins(%ln_out, %w_proj : memref<{S}x{D}xf32>, memref<{D}x{V}xf32>)
        outs(%logits : memref<{S}x{V}xf32>)

    return
  }}"""


def gen_full_mlir(S: int, D: int, FF: int, V: int) -> str:
    """Generate a single MLIR module containing all three kernel functions."""
    emb = gen_embedding_mlir(S, D, V)
    tb = gen_transformer_block_mlir(S, D, FF)
    lmh = gen_lm_head_mlir(S, D, V)

    return f"""\
// End-to-end GPT-2 inference (seq={S}, dim={D}, ff={FF}, vocab={V})
// Contains: @embedding, @transformer_block, @lm_head
module {{
{emb}

{tb}

{lmh}
}}
"""


# ---------------------------------------------------------------------------
# C wrapper generation
# ---------------------------------------------------------------------------

def gen_wrapper(S: int, D: int, FF: int, V: int, N: int) -> str:
    """Generate the C wrapper that chains embedding -> transformer_block x N -> lm_head."""

    # Build per-layer code
    layer_code_parts = []
    for i in range(N):
        layer_code_parts.append(f"""\
  /* ---- Layer {i}: unpack weights from static array ---- */
  {{
    const float *wp = layer{i}_weights;
    int off = 0;

    for (int j = 0; j < D; j++) tb_ln1_gamma[j] = wp[off++];
    for (int j = 0; j < D; j++) tb_ln1_beta[j]  = wp[off++];
    for (int j = 0; j < D * D; j++) tb_wq[j] = wp[off++];
    for (int j = 0; j < D * D; j++) tb_wk[j] = wp[off++];
    for (int j = 0; j < D * D; j++) tb_wv[j] = wp[off++];
    for (int j = 0; j < D * D; j++) tb_wo[j] = wp[off++];
    for (int j = 0; j < D; j++) tb_ln2_gamma[j] = wp[off++];
    for (int j = 0; j < D; j++) tb_ln2_beta[j]  = wp[off++];
    for (int j = 0; j < D * D_FF; j++) tb_w1[j] = wp[off++];
    for (int j = 0; j < D_FF * D; j++) tb_w2[j] = wp[off++];

    /* zero scratch buffers */
    for (int j = 0; j < S * D; j++) {{
      tb_x_ln[j] = 0; tb_q[j] = 0; tb_k[j] = 0; tb_v[j] = 0;
      tb_attn[j] = 0; tb_attn_out[j] = 0; tb_x_ln2[j] = 0;
      tb_x_out[j] = 0;
    }}
    for (int j = 0; j < S * S; j++) {{ tb_score[j] = 0; tb_prob[j] = 0; }}
    for (int j = 0; j < S * D_FF; j++) tb_hidden[j] = 0;
    for (int j = 0; j < S; j++) {{
      tb_ln_mean[j] = 0; tb_ln_var[j] = 0;
      tb_sm_max[j] = 0; tb_sm_sum[j] = 0;
    }}

    /* call transformer_block: x_cur -> tb_x_out */
    transformer_block(
        x_cur, tb_x_out,
        tb_ln1_gamma, tb_ln1_beta,
        tb_wq, tb_wk, tb_wv, tb_wo,
        tb_ln2_gamma, tb_ln2_beta,
        tb_w1, tb_w2,
        tb_x_ln, tb_q, tb_k, tb_v,
        tb_score, tb_prob, tb_attn, tb_attn_out,
        tb_x_ln2, tb_hidden,
        tb_ln_mean, tb_ln_var, tb_sm_max, tb_sm_sum);

    /* copy tb_x_out -> x_cur for next layer */
    for (int j = 0; j < S * D; j++)
      x_cur[j] = tb_x_out[j];
  }}""")

    layer_code = "\n\n".join(layer_code_parts)

    return f"""\
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

#include "full_inference_weights.h"

/* ---- Kernel declarations ---- */

extern void embedding(
    int *token_ids, float *tok_table, float *pos_table, float *output);

extern void transformer_block(
    float *x_in, float *x_out,
    float *ln1_gamma, float *ln1_beta,
    float *wq, float *wk, float *wv, float *wo,
    float *ln2_gamma, float *ln2_beta,
    float *w1, float *w2,
    float *x_ln, float *q, float *k, float *v,
    float *score, float *prob, float *attn, float *attn_out,
    float *x_ln2, float *hidden,
    float *ln_mean, float *ln_var, float *sm_max, float *sm_sum);

extern void lm_head(
    float *input, float *gamma, float *beta, float *w_proj,
    float *logits, float *ln_out, float *ln_mean, float *ln_var);

#define S     {S}
#define D     {D}
#define D_FF  {FF}
#define V     {V}

int main() {{
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  /* ==================================================================
   * Stage 1: Embedding
   * ================================================================== */

  int emb_token_ids[S];
  float emb_tok_table[V * D];
  float emb_pos_table[S * D];
  float x_cur[S * D];  /* current hidden state, reused across stages */

  /* Load embedding inputs from static const arrays */
  for (int i = 0; i < S; i++)
    emb_token_ids[i] = input_token_ids[i];
  for (int i = 0; i < V * D; i++)
    emb_tok_table[i] = tok_embed_table[i];
  for (int i = 0; i < S * D; i++)
    emb_pos_table[i] = pos_embed_table[i];

  embedding(emb_token_ids, emb_tok_table, emb_pos_table, x_cur);

  /* ==================================================================
   * Stage 2: Transformer blocks (x {N} layers)
   * ================================================================== */

  /* Transformer block weight buffers */
  float tb_ln1_gamma[D], tb_ln1_beta[D];
  float tb_wq[D * D], tb_wk[D * D], tb_wv[D * D], tb_wo[D * D];
  float tb_ln2_gamma[D], tb_ln2_beta[D];
  float tb_w1[D * D_FF], tb_w2[D_FF * D];

  /* Transformer block scratch buffers */
  float tb_x_out[S * D];
  float tb_x_ln[S * D];
  float tb_q[S * D], tb_k[S * D], tb_v[S * D];
  float tb_score[S * S], tb_prob[S * S];
  float tb_attn[S * D], tb_attn_out[S * D];
  float tb_x_ln2[S * D], tb_hidden[S * D_FF];
  float tb_ln_mean[S], tb_ln_var[S];
  float tb_sm_max[S], tb_sm_sum[S];

{layer_code}

  /* ==================================================================
   * Stage 3: LM Head (final LayerNorm + vocab projection)
   * ================================================================== */

  float lm_gamma[D], lm_beta[D];
  float lm_w_proj[D * V];
  float logits[S * V];
  float lm_ln_out[S * D];
  float lm_ln_mean[S], lm_ln_var[S];

  for (int i = 0; i < D; i++) {{
    lm_gamma[i] = final_ln_gamma[i];
    lm_beta[i]  = final_ln_beta[i];
  }}
  for (int i = 0; i < D * V; i++)
    lm_w_proj[i] = lm_head_proj[i];

  /* zero scratch */
  for (int i = 0; i < S * D; i++) lm_ln_out[i] = 0;
  for (int i = 0; i < S; i++) {{ lm_ln_mean[i] = 0; lm_ln_var[i] = 0; }}
  for (int i = 0; i < S * V; i++) logits[i] = 0;

  lm_head(x_cur, lm_gamma, lm_beta, lm_w_proj,
          logits, lm_ln_out, lm_ln_mean, lm_ln_var);

  /* ==================================================================
   * Verify: compare logits against golden
   * ================================================================== */

  int pass = 1;
  float max_diff = 0.0f;
  for (int i = 0; i < S * V; i++) {{
    float diff = logits[i] - golden_logits[i];
    if (diff < 0) diff = -diff;
    if (diff > max_diff) max_diff = diff;
    if (diff > 5e-2f) {{
      pass = 0;
    }}
  }}

  if (pass) {{
    vx_printf("full_inference PASSED (max_diff=%d.%04d)\\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
    return 0;
  }} else {{
    vx_printf("full_inference FAILED (max_diff=%d.%04d)\\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
    return 1;
  }}
}}
"""


# ---------------------------------------------------------------------------
# Weights header generation
# ---------------------------------------------------------------------------

def gen_weights_header(token_ids: np.ndarray, tok_table: np.ndarray,
                       pos_table: np.ndarray,
                       layers_weights: list,
                       final_ln_gamma: np.ndarray, final_ln_beta: np.ndarray,
                       lm_head_w: np.ndarray,
                       golden_logits_arr: np.ndarray,
                       S: int, D: int, FF: int, V: int) -> str:
    parts = []
    parts.append("/* Auto-generated by gen_full_inference.py -- do not edit */")
    parts.append("#ifndef FULL_INFERENCE_WEIGHTS_H")
    parts.append("#define FULL_INFERENCE_WEIGHTS_H")
    parts.append("")

    # Input token IDs
    parts.append(f"/* Input token IDs: [{S}] */")
    parts.append(f"static const int input_token_ids[{S}] = {{")
    parts.append(int_array_to_c_initializer(token_ids))
    parts.append("};")
    parts.append("")

    # Token embedding table
    parts.append(f"/* Token embedding table: [{V}, {D}] = {V * D} floats */")
    parts.append(f"static const float tok_embed_table[{V * D}] = {{")
    parts.append(array_to_c_initializer(tok_table))
    parts.append("};")
    parts.append("")

    # Position embedding table
    parts.append(f"/* Position embedding table: [{S}, {D}] = {S * D} floats */")
    parts.append(f"static const float pos_embed_table[{S * D}] = {{")
    parts.append(array_to_c_initializer(pos_table))
    parts.append("};")
    parts.append("")

    # Per-layer weights
    for i, weights in enumerate(layers_weights):
        all_data = []
        for key in WEIGHT_ORDER:
            all_data.append(weights[key].ravel())
        concat = np.concatenate(all_data)
        total = len(concat)

        parts.append(f"/* Layer {i}: {total} floats */")
        parts.append(f"static const float layer{i}_weights[{total}] = {{")
        parts.append(array_to_c_initializer(concat))
        parts.append("};")
        parts.append("")

    # Final LayerNorm
    parts.append(f"/* Final LayerNorm gamma: [{D}] */")
    parts.append(f"static const float final_ln_gamma[{D}] = {{")
    parts.append(array_to_c_initializer(final_ln_gamma))
    parts.append("};")
    parts.append("")

    parts.append(f"/* Final LayerNorm beta: [{D}] */")
    parts.append(f"static const float final_ln_beta[{D}] = {{")
    parts.append(array_to_c_initializer(final_ln_beta))
    parts.append("};")
    parts.append("")

    # LM head projection
    parts.append(f"/* LM head projection: [{D}, {V}] = {D * V} floats */")
    parts.append(f"static const float lm_head_proj[{D * V}] = {{")
    parts.append(array_to_c_initializer(lm_head_w))
    parts.append("};")
    parts.append("")

    # Golden logits
    golden_flat = golden_logits_arr.ravel().astype(np.float32)
    parts.append(f"/* Golden logits: [{S}, {V}] = {len(golden_flat)} floats */")
    parts.append(f"static const float golden_logits[{len(golden_flat)}] = {{")
    parts.append(array_to_c_initializer(golden_flat))
    parts.append("};")
    parts.append("")

    parts.append("#endif /* FULL_INFERENCE_WEIGHTS_H */")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate end-to-end GPT-2 inference MLIR, C wrapper, and weights"
    )
    parser.add_argument("--seq", type=int, required=True, help="Sequence length (S)")
    parser.add_argument("--dim", type=int, required=True, help="Model dimension (D)")
    parser.add_argument("--ff", type=int, required=True, help="Feed-forward dimension (FF)")
    parser.add_argument("--vocab", type=int, required=True, help="Vocabulary size (V)")
    parser.add_argument("--layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    S = args.seq
    D = args.dim
    FF = args.ff
    V = args.vocab
    N = args.layers
    seed = args.seed
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    # ---- Generate all weights ----

    # Token embedding table: [V, D]
    tok_table = (0.02 * rng.randn(V, D)).astype(np.float32)

    # Position embedding table: [S, D]
    pos_table = (0.02 * rng.randn(S, D)).astype(np.float32)

    # Transformer block weights per layer
    all_layers_weights = []
    for i in range(N):
        w = generate_layer_weights(D, FF, rng)
        all_layers_weights.append(w)

    # Final LayerNorm
    final_ln_gamma = np.ones(D, dtype=np.float32)
    final_ln_beta = np.zeros(D, dtype=np.float32)

    # LM head projection: [D, V]
    lm_head_w = (0.02 * rng.randn(D, V)).astype(np.float32)

    # ---- Generate random token IDs ----
    token_ids = rng.randint(0, V, size=S).astype(np.int32)

    # ---- Run numpy reference: full pipeline ----

    # 1. Embedding
    x = embedding_ref(token_ids, tok_table, pos_table)
    x = x.astype(np.float32)

    # 2. Transformer blocks
    for i in range(N):
        x = transformer_block(x, all_layers_weights[i])
        x = x.astype(np.float32)

    # 3. LM head
    golden_logits_arr = lm_head_ref(x, final_ln_gamma, final_ln_beta, lm_head_w)
    golden_logits_arr = golden_logits_arr.astype(np.float32)

    # ---- Generate output files ----

    # 1. MLIR
    mlir_content = gen_full_mlir(S, D, FF, V)
    mlir_path = os.path.join(out_dir, "full_inference.mlir")
    with open(mlir_path, "w") as f:
        f.write(mlir_content)
    print(f"Wrote {mlir_path}")

    # 2. Weights header
    header_content = gen_weights_header(
        token_ids, tok_table, pos_table,
        all_layers_weights,
        final_ln_gamma, final_ln_beta,
        lm_head_w, golden_logits_arr,
        S, D, FF, V
    )
    header_path = os.path.join(out_dir, "full_inference_weights.h")
    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"Wrote {header_path}")

    # 3. C wrapper
    wrapper_content = gen_wrapper(S, D, FF, V, N)
    wrapper_path = os.path.join(out_dir, "full_inference_wrapper.c")
    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)
    print(f"Wrote {wrapper_path}")

    # Print summary
    layer_weights_count = (2 * D + 4 * D * D + 2 * D + D * FF + FF * D)
    total_weights = (V * D + S * D + N * layer_weights_count + D + D + D * V)
    print(f"\nSummary: seq={S}, dim={D}, ff={FF}, vocab={V}, layers={N}, seed={seed}")
    print(f"  Token embedding table: {V * D} floats")
    print(f"  Position embedding table: {S * D} floats")
    print(f"  Weights per transformer layer: {layer_weights_count} floats")
    print(f"  Final LN: {2 * D} floats")
    print(f"  LM head projection: {D * V} floats")
    print(f"  Total weight floats: {total_weights}")
    print(f"  Golden logits: [{S}, {V}] = {S * V} floats")
    print(f"  Golden logits sample (first 8): {golden_logits_arr.ravel()[:8]}")


if __name__ == "__main__":
    main()
