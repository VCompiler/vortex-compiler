// Single-head attention (seq=4, d=8)
// Q = x @ Wq, K = x @ Wk, V = x @ Wv
// score = Q @ K^T / sqrt(d)
// prob = softmax(score)
// attn = prob @ V
// out = attn @ Wo
module {
  func.func @attention(
      %x: memref<4x8xf32>,
      %wq: memref<8x8xf32>, %wk: memref<8x8xf32>,
      %wv: memref<8x8xf32>, %wo: memref<8x8xf32>,
      %q: memref<4x8xf32>, %k: memref<4x8xf32>, %v: memref<4x8xf32>,
      %score: memref<4x4xf32>, %prob: memref<4x4xf32>,
      %attn: memref<4x8xf32>, %out: memref<4x8xf32>,
      %sm_max: memref<4xf32>, %sm_sum: memref<4xf32>)
      attributes {vortex.entry} {

    %zero = arith.constant 0.0 : f32
    %scale = arith.constant 0.35355339059 : f32  // 1/sqrt(8)
    %neg_inf = arith.constant 0xFF800000 : f32

    // Q = x @ Wq
    linalg.fill ins(%zero : f32) outs(%q : memref<4x8xf32>)
    linalg.matmul ins(%x, %wq : memref<4x8xf32>, memref<8x8xf32>)
        outs(%q : memref<4x8xf32>)

    // K = x @ Wk
    linalg.fill ins(%zero : f32) outs(%k : memref<4x8xf32>)
    linalg.matmul ins(%x, %wk : memref<4x8xf32>, memref<8x8xf32>)
        outs(%k : memref<4x8xf32>)

    // V = x @ Wv
    linalg.fill ins(%zero : f32) outs(%v : memref<4x8xf32>)
    linalg.matmul ins(%x, %wv : memref<4x8xf32>, memref<8x8xf32>)
        outs(%v : memref<4x8xf32>)

    // score = Q @ K^T / sqrt(d)
    // K^T via linalg.generic with transposed indexing_maps
    linalg.fill ins(%zero : f32) outs(%score : memref<4x4xf32>)
    linalg.generic {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,   // Q[m, k]
        affine_map<(m, n, k) -> (n, k)>,   // K[n, k] (transposed: K^T[k, n] -> access K[n, k])
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
    // scale by 1/sqrt(d)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%score : memref<4x4xf32>) outs(%score : memref<4x4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %scaled = arith.mulf %s, %scale : f32
      linalg.yield %scaled : f32
    }

    // prob = softmax(score) along axis=-1
    // Step 1: reduce_max
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
    // Step 2: exp(score - max)
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
    // Step 3: reduce_sum
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
    // Step 4: div
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

    // attn = prob @ V
    linalg.fill ins(%zero : f32) outs(%attn : memref<4x8xf32>)
    linalg.matmul ins(%prob, %v : memref<4x4xf32>, memref<4x8xf32>)
        outs(%attn : memref<4x8xf32>)

    // out = attn @ Wo
    linalg.fill ins(%zero : f32) outs(%out : memref<4x8xf32>)
    linalg.matmul ins(%attn, %wo : memref<4x8xf32>, memref<8x8xf32>)
        outs(%out : memref<4x8xf32>)

    return
  }
}
