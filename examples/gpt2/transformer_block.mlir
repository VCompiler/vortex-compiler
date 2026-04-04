// Complete single-head Transformer block (seq=4, d_model=8, d_ff=32)
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
module {
  func.func @transformer_block(
      // Input / output
      %x_in: memref<4x8xf32>,
      %x_out: memref<4x8xf32>,
      // LayerNorm 1 weights
      %ln1_gamma: memref<8xf32>,
      %ln1_beta: memref<8xf32>,
      // Attention weights
      %wq: memref<8x8xf32>,
      %wk: memref<8x8xf32>,
      %wv: memref<8x8xf32>,
      %wo: memref<8x8xf32>,
      // LayerNorm 2 weights
      %ln2_gamma: memref<8xf32>,
      %ln2_beta: memref<8xf32>,
      // MLP weights
      %w1: memref<8x32xf32>,
      %w2: memref<32x8xf32>,
      // Scratch buffers
      %x_ln: memref<4x8xf32>,
      %q: memref<4x8xf32>,
      %k: memref<4x8xf32>,
      %v: memref<4x8xf32>,
      %score: memref<4x4xf32>,
      %prob: memref<4x4xf32>,
      %attn: memref<4x8xf32>,
      %attn_out: memref<4x8xf32>,
      %x_ln2: memref<4x8xf32>,
      %hidden: memref<4x32xf32>,
      %ln_mean: memref<4xf32>,
      %ln_var: memref<4xf32>,
      %sm_max: memref<4xf32>,
      %sm_sum: memref<4xf32>)
      attributes {vortex.entry} {

    // ---- Constants ----
    %zero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %inv_n = arith.constant 0.125 : f32       // 1/8  (d_model)
    %scale = arith.constant 0.35355339059 : f32 // 1/sqrt(8)
    %neg_inf = arith.constant 0xFF800000 : f32 // -inf
    %half = arith.constant 0.5 : f32
    %inv_sqrt2 = arith.constant 0.70710678118 : f32
    %one = arith.constant 1.0 : f32

    // ================================================================
    // Step 1: LayerNorm(x_in, ln1_gamma, ln1_beta) -> x_ln
    // ================================================================

    // 1a: mean = sum(x_in) / N
    linalg.fill ins(%zero : f32) outs(%ln_mean : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%x_in : memref<4x8xf32>) outs(%ln_mean : memref<4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%ln_mean : memref<4xf32>) outs(%ln_mean : memref<4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %m = arith.mulf %s, %inv_n : f32
      linalg.yield %m : f32
    }

    // 1b: var = sum((x_in - mean)^2) / N
    linalg.fill ins(%zero : f32) outs(%ln_var : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%x_in, %ln_mean : memref<4x8xf32>, memref<4xf32>)
      outs(%ln_var : memref<4xf32>) {
    ^bb0(%x: f32, %mean: f32, %acc: f32):
      %diff = arith.subf %x, %mean : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %acc : f32
      linalg.yield %s : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%ln_var : memref<4xf32>) outs(%ln_var : memref<4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %vr = arith.mulf %s, %inv_n : f32
      linalg.yield %vr : f32
    }

    // 1c: x_ln = (x_in - mean) / sqrt(var + eps) * gamma + beta
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x_in, %ln_mean, %ln_var, %ln1_gamma, %ln1_beta :
          memref<4x8xf32>, memref<4xf32>, memref<4xf32>,
          memref<8xf32>, memref<8xf32>)
      outs(%x_ln : memref<4x8xf32>) {
    ^bb0(%x: f32, %mean: f32, %var: f32, %g: f32, %b: f32, %dummy: f32):
      %diff = arith.subf %x, %mean : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %normed = arith.mulf %diff, %inv_std : f32
      %scaled = arith.mulf %normed, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    }

    // ================================================================
    // Step 2: Q = x_ln @ Wq, K = x_ln @ Wk, V = x_ln @ Wv
    // ================================================================

    // Q = x_ln @ Wq
    linalg.fill ins(%zero : f32) outs(%q : memref<4x8xf32>)
    linalg.matmul ins(%x_ln, %wq : memref<4x8xf32>, memref<8x8xf32>)
        outs(%q : memref<4x8xf32>)

    // K = x_ln @ Wk
    linalg.fill ins(%zero : f32) outs(%k : memref<4x8xf32>)
    linalg.matmul ins(%x_ln, %wk : memref<4x8xf32>, memref<8x8xf32>)
        outs(%k : memref<4x8xf32>)

    // V = x_ln @ Wv
    linalg.fill ins(%zero : f32) outs(%v : memref<4x8xf32>)
    linalg.matmul ins(%x_ln, %wv : memref<4x8xf32>, memref<8x8xf32>)
        outs(%v : memref<4x8xf32>)

    // ================================================================
    // Step 3: score = Q @ K^T / sqrt(d)
    // ================================================================

    // Q @ K^T via linalg.generic with transposed indexing for K
    linalg.fill ins(%zero : f32) outs(%score : memref<4x4xf32>)
    linalg.generic {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,   // Q[m, k]
        affine_map<(m, n, k) -> (n, k)>,   // K[n, k]  (= K^T[k, n])
        affine_map<(m, n, k) -> (m, n)>    // score[m, n]
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%q, %k : memref<4x8xf32>, memref<4x8xf32>)
      outs(%score : memref<4x4xf32>) {
    ^bb0(%q_val: f32, %k_val: f32, %acc: f32):
      %prod = arith.mulf %q_val, %k_val : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
    }

    // Scale by 1/sqrt(d)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%score : memref<4x4xf32>) outs(%score : memref<4x4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %scaled = arith.mulf %s, %scale : f32
      linalg.yield %scaled : f32
    }

    // ================================================================
    // Step 4: prob = softmax(score) along axis=-1
    // ================================================================

    // 4a: reduce_max
    linalg.fill ins(%neg_inf : f32) outs(%sm_max : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%score : memref<4x4xf32>) outs(%sm_max : memref<4xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %mx = arith.maximumf %a, %acc : f32
      linalg.yield %mx : f32
    }

    // 4b: exp(score - max) -> prob
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%score, %sm_max : memref<4x4xf32>, memref<4xf32>)
      outs(%prob : memref<4x4xf32>) {
    ^bb0(%s: f32, %mx: f32, %dummy: f32):
      %shifted = arith.subf %s, %mx : f32
      %e = math.exp %shifted : f32
      linalg.yield %e : f32
    }

    // 4c: reduce_sum
    linalg.fill ins(%zero : f32) outs(%sm_sum : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%prob : memref<4x4xf32>) outs(%sm_sum : memref<4xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %s = arith.addf %a, %acc : f32
      linalg.yield %s : f32
    }

    // 4d: div by sum
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%prob, %sm_sum : memref<4x4xf32>, memref<4xf32>)
      outs(%prob : memref<4x4xf32>) {
    ^bb0(%e: f32, %s: f32, %dummy: f32):
      %r = arith.divf %e, %s : f32
      linalg.yield %r : f32
    }

    // ================================================================
    // Step 5: attn = prob @ V
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%attn : memref<4x8xf32>)
    linalg.matmul ins(%prob, %v : memref<4x4xf32>, memref<4x8xf32>)
        outs(%attn : memref<4x8xf32>)

    // ================================================================
    // Step 6: attn_out = attn @ Wo
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%attn_out : memref<4x8xf32>)
    linalg.matmul ins(%attn, %wo : memref<4x8xf32>, memref<8x8xf32>)
        outs(%attn_out : memref<4x8xf32>)

    // ================================================================
    // Step 7: x_in = x_in + attn_out  (residual add, in-place)
    // ================================================================

    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x_in, %attn_out : memref<4x8xf32>, memref<4x8xf32>)
      outs(%x_in : memref<4x8xf32>) {
    ^bb0(%a: f32, %b: f32, %dummy: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }

    // ================================================================
    // Step 8: LayerNorm(x_in, ln2_gamma, ln2_beta) -> x_ln2
    // ================================================================

    // 8a: mean = sum(x_in) / N
    linalg.fill ins(%zero : f32) outs(%ln_mean : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%x_in : memref<4x8xf32>) outs(%ln_mean : memref<4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%ln_mean : memref<4xf32>) outs(%ln_mean : memref<4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %m = arith.mulf %s, %inv_n : f32
      linalg.yield %m : f32
    }

    // 8b: var = sum((x_in - mean)^2) / N
    linalg.fill ins(%zero : f32) outs(%ln_var : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%x_in, %ln_mean : memref<4x8xf32>, memref<4xf32>)
      outs(%ln_var : memref<4xf32>) {
    ^bb0(%x: f32, %mean: f32, %acc: f32):
      %diff = arith.subf %x, %mean : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %acc : f32
      linalg.yield %s : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%ln_var : memref<4xf32>) outs(%ln_var : memref<4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %vr = arith.mulf %s, %inv_n : f32
      linalg.yield %vr : f32
    }

    // 8c: x_ln2 = (x_in - mean) / sqrt(var + eps) * gamma + beta
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x_in, %ln_mean, %ln_var, %ln2_gamma, %ln2_beta :
          memref<4x8xf32>, memref<4xf32>, memref<4xf32>,
          memref<8xf32>, memref<8xf32>)
      outs(%x_ln2 : memref<4x8xf32>) {
    ^bb0(%x: f32, %mean: f32, %var: f32, %g: f32, %b: f32, %dummy: f32):
      %diff = arith.subf %x, %mean : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %normed = arith.mulf %diff, %inv_std : f32
      %scaled = arith.mulf %normed, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    }

    // ================================================================
    // Step 9: hidden = x_ln2 @ W1  (d_model -> d_ff)
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%hidden : memref<4x32xf32>)
    linalg.matmul ins(%x_ln2, %w1 : memref<4x8xf32>, memref<8x32xf32>)
        outs(%hidden : memref<4x32xf32>)

    // ================================================================
    // Step 10: hidden = GeLU(hidden)
    // ================================================================

    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%hidden : memref<4x32xf32>) outs(%hidden : memref<4x32xf32>) {
    ^bb0(%x: f32, %dummy: f32):
      %x_sc = arith.mulf %x, %inv_sqrt2 : f32
      %erf_val = math.erf %x_sc : f32
      %one_erf = arith.addf %one, %erf_val : f32
      %x_half = arith.mulf %x, %half : f32
      %result = arith.mulf %x_half, %one_erf : f32
      linalg.yield %result : f32
    }

    // ================================================================
    // Step 11: attn_out = hidden @ W2  (d_ff -> d_model)
    //   (reuse attn_out as scratch for MLP output)
    // ================================================================

    linalg.fill ins(%zero : f32) outs(%attn_out : memref<4x8xf32>)
    linalg.matmul ins(%hidden, %w2 : memref<4x32xf32>, memref<32x8xf32>)
        outs(%attn_out : memref<4x8xf32>)

    // ================================================================
    // Step 12: x_out = x_in + attn_out  (residual add, store to output)
    //   x_in already contains x + attention_residual from Step 7
    // ================================================================

    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x_in, %attn_out : memref<4x8xf32>, memref<4x8xf32>)
      outs(%x_out : memref<4x8xf32>) {
    ^bb0(%a: f32, %b: f32, %dummy: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    }

    return
  }
}
