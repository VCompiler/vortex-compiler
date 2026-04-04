#!/usr/bin/env python3
"""Generate parameterized transformer block MLIR and C wrapper files.

Usage:
    python3 gen_transformer.py --seq S --dim D --ff FF --out-dir DIR [--name NAME]

Produces:
    DIR/<NAME>.mlir
    DIR/<NAME>_wrapper.c
"""

import argparse
import math
import os
import sys


def gen_mlir(S: int, D: int, FF: int, func_name: str) -> str:
    inv_n = 1.0 / D
    scale = 1.0 / math.sqrt(D)

    # Format constants to sufficient precision
    inv_n_str = f"{inv_n:.17g}"
    scale_str = f"{scale:.11f}"

    return f"""\
// Complete single-head Transformer block (seq={S}, d_model={D}, d_ff={FF})
// 1. LayerNorm(x) -> x_ln
// 2. Q = x_ln @ Wq, K = x_ln @ Wk, V = x_ln @ Wv
// 3. score = Q @ K^T / sqrt(d)
// 4. prob = softmax(score)
// 5. attn = prob @ V
// 6. out1 = attn @ Wo
// 7. x = x + out1  (residual)
// 8. LayerNorm(x) -> x_ln2
// 9. h = x_ln2 @ W1, GeLU(h), h @ W2
// 10. x_out = x + h  (residual)
module {{
  func.func @{func_name}(
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
    //   (reuse attn_out as scratch for MLP output)
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%attn_out : memref<{S}x{D}xf32>)
    linalg.matmul ins(%hidden, %w2 : memref<{S}x{FF}xf32>, memref<{FF}x{D}xf32>)
        outs(%attn_out : memref<{S}x{D}xf32>)

    // ================================================================
    // Step 12: x_out = x_in + attn_out  (residual add, store to output)
    //   x_in already contains x + attention_residual from Step 7
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
  }}
}}
"""


def gen_wrapper(S: int, D: int, FF: int, func_name: str) -> str:
    return f"""\
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void {func_name}(
    float *x_in, float *x_out,
    float *ln1_gamma, float *ln1_beta,
    float *wq, float *wk, float *wv, float *wo,
    float *ln2_gamma, float *ln2_beta,
    float *w1, float *w2,
    float *x_ln, float *q, float *k, float *v,
    float *score, float *prob, float *attn, float *attn_out,
    float *x_ln2, float *hidden,
    float *ln_mean, float *ln_var, float *sm_max, float *sm_sum);

#define S     {S}
#define D     {D}
#define D_FF  {FF}
#define EPS   1e-5f

/* ---- reference helpers ---- */

static void layernorm_ref(const float *x, const float *gamma,
                          const float *beta, float *out,
                          int rows, int cols, float eps) {{
  for (int r = 0; r < rows; r++) {{
    float mean = 0.0f;
    for (int c = 0; c < cols; c++)
      mean += x[r * cols + c];
    mean /= cols;

    float var = 0.0f;
    for (int c = 0; c < cols; c++) {{
      float d = x[r * cols + c] - mean;
      var += d * d;
    }}
    var /= cols;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (int c = 0; c < cols; c++)
      out[r * cols + c] =
          (x[r * cols + c] - mean) * inv_std * gamma[c] + beta[c];
  }}
}}

static void matmul_ref(const float *a, const float *b, float *c,
                       int m, int n, int k_) {{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {{
      float acc = 0.0f;
      for (int kk = 0; kk < k_; kk++)
        acc += a[i * k_ + kk] * b[kk * n + j];
      c[i * n + j] = acc;
    }}
}}

static float gelu_ref(float x) {{
  return x * 0.5f * (1.0f + erff(x * 0.70710678118f));
}}

static void softmax_ref(const float *in, float *out, int rows, int cols) {{
  for (int r = 0; r < rows; r++) {{
    float mx = in[r * cols];
    for (int c = 1; c < cols; c++)
      if (in[r * cols + c] > mx)
        mx = in[r * cols + c];
    float s = 0.0f;
    for (int c = 0; c < cols; c++) {{
      out[r * cols + c] = expf(in[r * cols + c] - mx);
      s += out[r * cols + c];
    }}
    for (int c = 0; c < cols; c++)
      out[r * cols + c] /= s;
  }}
}}

int main() {{
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  /* ---- allocate all buffers on the stack ---- */
  float x_in[S * D], x_out[S * D];
  float ln1_gamma[D], ln1_beta[D];
  float wq[D * D], wk[D * D], wv[D * D], wo[D * D];
  float ln2_gamma[D], ln2_beta[D];
  float w1[D * D_FF], w2[D_FF * D];

  /* scratch */
  float x_ln[S * D];
  float q[S * D], k[S * D], v[S * D];
  float score[S * S], prob[S * S];
  float attn[S * D], attn_out[S * D];
  float x_ln2[S * D], hidden[S * D_FF];
  float ln_mean[S], ln_var[S];
  float sm_max[S], sm_sum[S];

  /* ---- deterministic initialisation ---- */
  for (int i = 0; i < S * D; i++)
    x_in[i] = (float)(i % 5) * 0.1f - 0.2f;

  for (int i = 0; i < D; i++) {{
    ln1_gamma[i] = 1.0f;
    ln1_beta[i]  = 0.0f;
    ln2_gamma[i] = 1.0f;
    ln2_beta[i]  = 0.0f;
  }}

  for (int i = 0; i < D * D; i++) {{
    wq[i] = (float)(i % 7)  * 0.02f - 0.06f;
    wk[i] = (float)(i % 9)  * 0.02f - 0.08f;
    wv[i] = (float)(i % 11) * 0.02f - 0.10f;
    wo[i] = (float)(i % 13) * 0.02f - 0.12f;
  }}

  for (int i = 0; i < D * D_FF; i++)
    w1[i] = (float)(i % 17) * 0.02f - 0.16f;

  for (int i = 0; i < D_FF * D; i++)
    w2[i] = (float)(i % 19) * 0.02f - 0.18f;

  /* zero all scratch buffers */
  for (int i = 0; i < S * D; i++) {{
    x_ln[i] = 0.0f; q[i] = 0.0f; k[i] = 0.0f; v[i] = 0.0f;
    attn[i] = 0.0f; attn_out[i] = 0.0f; x_ln2[i] = 0.0f;
    x_out[i] = 0.0f;
  }}
  for (int i = 0; i < S * S; i++) {{ score[i] = 0.0f; prob[i] = 0.0f; }}
  for (int i = 0; i < S * D_FF; i++) hidden[i] = 0.0f;
  for (int i = 0; i < S; i++) {{
    ln_mean[i] = 0.0f; ln_var[i] = 0.0f;
    sm_max[i]  = 0.0f; sm_sum[i] = 0.0f;
  }}

  /* ---- call kernel ---- */
  {func_name}(
      x_in, x_out,
      ln1_gamma, ln1_beta,
      wq, wk, wv, wo,
      ln2_gamma, ln2_beta,
      w1, w2,
      x_ln, q, k, v,
      score, prob, attn, attn_out,
      x_ln2, hidden,
      ln_mean, ln_var, sm_max, sm_sum);

  /* ---- CPU reference ---- */

  /* 1. LayerNorm1 */
  float r_xln[S * D];
  layernorm_ref(x_in, ln1_gamma, ln1_beta, r_xln, S, D, EPS);

  /* 2. Q = x_ln @ Wq,  K = x_ln @ Wk,  V = x_ln @ Wv */
  float r_q[S * D], r_k[S * D], r_v[S * D];
  matmul_ref(r_xln, wq, r_q, S, D, D);
  matmul_ref(r_xln, wk, r_k, S, D, D);
  matmul_ref(r_xln, wv, r_v, S, D, D);

  /* 3. score = Q @ K^T / sqrt(D) */
  float r_score[S * S];
  for (int i = 0; i < S; i++)
    for (int j = 0; j < S; j++) {{
      float acc = 0.0f;
      for (int kk = 0; kk < D; kk++)
        acc += r_q[i * D + kk] * r_k[j * D + kk]; /* K^T */
      r_score[i * S + j] = acc * (1.0f / sqrtf((float)D));
    }}

  /* 4. prob = softmax(score) */
  float r_prob[S * S];
  softmax_ref(r_score, r_prob, S, S);

  /* 5. attn = prob @ V */
  float r_attn[S * D];
  matmul_ref(r_prob, r_v, r_attn, S, D, S);

  /* 6. attn_out = attn @ Wo */
  float r_attn_out[S * D];
  matmul_ref(r_attn, wo, r_attn_out, S, D, D);

  /* 7. residual1 = x_in + attn_out */
  float r_res1[S * D];
  for (int i = 0; i < S * D; i++)
    r_res1[i] = x_in[i] + r_attn_out[i];

  /* 8. LayerNorm2 */
  float r_xln2[S * D];
  layernorm_ref(r_res1, ln2_gamma, ln2_beta, r_xln2, S, D, EPS);

  /* 9. hidden = GELU(x_ln2 @ W1) */
  float r_hidden[S * D_FF];
  matmul_ref(r_xln2, w1, r_hidden, S, D_FF, D);
  for (int i = 0; i < S * D_FF; i++)
    r_hidden[i] = gelu_ref(r_hidden[i]);

  /* 10. ff_out = hidden @ W2 */
  float r_ff_out[S * D];
  matmul_ref(r_hidden, w2, r_ff_out, S, D, D_FF);

  /* 11. x_out = residual1 + ff_out */
  float r_xout[S * D];
  for (int i = 0; i < S * D; i++)
    r_xout[i] = r_res1[i] + r_ff_out[i];

  /* ---- compare ---- */
  int pass = 1;
  float max_diff = 0.0f;
  for (int i = 0; i < S * D; i++) {{
    float diff = x_out[i] - r_xout[i];
    if (diff < 0) diff = -diff;
    if (diff > max_diff) max_diff = diff;
    if (diff > 5e-2f) {{
      pass = 0;
    }}
  }}

  if (pass) {{
    vx_printf("{func_name} passed (max_diff=%d.%04d)\\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
    return 0;
  }} else {{
    vx_printf("{func_name} FAILED (max_diff=%d.%04d)\\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
    return 1;
  }}
}}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameterized transformer block MLIR and C wrapper"
    )
    parser.add_argument("--seq", type=int, required=True, help="Sequence length (S)")
    parser.add_argument("--dim", type=int, required=True, help="Model dimension (D)")
    parser.add_argument("--ff", type=int, required=True, help="Feed-forward dimension (FF)")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Kernel function name and file prefix (default: transformer_block_{S}x{D}x{FF})",
    )
    args = parser.parse_args()

    S = args.seq
    D = args.dim
    FF = args.ff
    out_dir = args.out_dir
    name = args.name if args.name else f"transformer_block_{S}x{D}x{FF}"

    os.makedirs(out_dir, exist_ok=True)

    mlir_path = os.path.join(out_dir, f"{name}.mlir")
    wrapper_path = os.path.join(out_dir, f"{name}_wrapper.c")

    # The MLIR function is always called "transformer_block" to match the extern
    # declaration pattern. The --name controls only the file prefix.
    func_name = "transformer_block"

    mlir_content = gen_mlir(S, D, FF, func_name)
    wrapper_content = gen_wrapper(S, D, FF, func_name)

    with open(mlir_path, "w") as f:
        f.write(mlir_content)

    with open(wrapper_path, "w") as f:
        f.write(wrapper_content)

    print(f"Generated {mlir_path}")
    print(f"Generated {wrapper_path}")


if __name__ == "__main__":
    main()
