// LM Head: final LayerNorm + vocab projection
// input:   memref<32x64xf32>   (last block output)
// gamma:   memref<64xf32>      (final LN gamma)
// beta:    memref<64xf32>      (final LN beta)
// w_proj:  memref<64x256xf32>  (vocab projection, vocab=256)
// logits:  memref<32x256xf32>  (output logits)
// ln_out:  memref<32x64xf32>   (scratch)
// ln_mean: memref<32xf32>      (scratch)
// ln_var:  memref<32xf32>      (scratch)
module {
  func.func @lm_head(%input: memref<32x64xf32>,
                     %gamma: memref<64xf32>, %beta: memref<64xf32>,
                     %w_proj: memref<64x256xf32>,
                     %logits: memref<32x256xf32>,
                     %ln_out: memref<32x64xf32>,
                     %ln_mean: memref<32xf32>, %ln_var: memref<32xf32>)
      attributes {vortex.entry} {

    %zero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %inv_n = arith.constant 0.015625 : f32  // 1/64

    // LayerNorm: mean
    linalg.fill ins(%zero : f32) outs(%ln_mean : memref<32xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : memref<32x64xf32>) outs(%ln_mean : memref<32xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%ln_mean : memref<32xf32>) outs(%ln_mean : memref<32xf32>) {
    ^bb0(%s: f32, %d: f32):
      %m = arith.mulf %s, %inv_n : f32
      linalg.yield %m : f32
    }

    // LayerNorm: variance
    linalg.fill ins(%zero : f32) outs(%ln_var : memref<32xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%input, %ln_mean : memref<32x64xf32>, memref<32xf32>)
      outs(%ln_var : memref<32xf32>) {
    ^bb0(%x: f32, %mean: f32, %acc: f32):
      %diff = arith.subf %x, %mean : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %acc : f32
      linalg.yield %s : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%ln_var : memref<32xf32>) outs(%ln_var : memref<32xf32>) {
    ^bb0(%s: f32, %d: f32):
      %v = arith.mulf %s, %inv_n : f32
      linalg.yield %v : f32
    }

    // LayerNorm: normalize
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %ln_mean, %ln_var, %gamma, %beta :
          memref<32x64xf32>, memref<32xf32>, memref<32xf32>,
          memref<64xf32>, memref<64xf32>)
      outs(%ln_out : memref<32x64xf32>) {
    ^bb0(%x: f32, %mean: f32, %var: f32, %g: f32, %b: f32, %dummy: f32):
      %diff = arith.subf %x, %mean : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %normed = arith.mulf %diff, %inv_std : f32
      %scaled = arith.mulf %normed, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    }

    // Vocab projection: logits = ln_out @ w_proj
    linalg.fill ins(%zero : f32) outs(%logits : memref<32x256xf32>)
    linalg.matmul ins(%ln_out, %w_proj : memref<32x64xf32>, memref<64x256xf32>)
        outs(%logits : memref<32x256xf32>)

    return
  }
}
