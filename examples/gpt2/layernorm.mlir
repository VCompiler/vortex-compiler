// LayerNorm: normalize along last axis with gamma/beta
// input:  memref<4x8xf32>
// gamma:  memref<8xf32>
// beta:   memref<8xf32>
// output: memref<4x8xf32>
// tmp_mean, tmp_var: memref<4xf32> (scratch)
module {
  func.func @layernorm(%in: memref<4x8xf32>, %gamma: memref<8xf32>,
                       %beta: memref<8xf32>, %out: memref<4x8xf32>,
                       %tmp_mean: memref<4xf32>, %tmp_var: memref<4xf32>)
      attributes {vortex.entry} {

    %zero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-5 : f32
    %inv_n = arith.constant 0.125 : f32  // 1.0 / 8

    // Step 1: compute mean = sum(x) / N
    linalg.fill ins(%zero : f32) outs(%tmp_mean : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%in : memref<4x8xf32>) outs(%tmp_mean : memref<4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
    }
    // divide by N to get mean
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%tmp_mean : memref<4xf32>) outs(%tmp_mean : memref<4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %m = arith.mulf %s, %inv_n : f32
      linalg.yield %m : f32
    }

    // Step 2: compute variance = sum((x - mean)^2) / N
    linalg.fill ins(%zero : f32) outs(%tmp_var : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%in, %tmp_mean : memref<4x8xf32>, memref<4xf32>)
      outs(%tmp_var : memref<4xf32>) {
    ^bb0(%x: f32, %mean: f32, %acc: f32):
      %diff = arith.subf %x, %mean : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %acc : f32
      linalg.yield %s : f32
    }
    // divide by N
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%tmp_var : memref<4xf32>) outs(%tmp_var : memref<4xf32>) {
    ^bb0(%s: f32, %dummy: f32):
      %v = arith.mulf %s, %inv_n : f32
      linalg.yield %v : f32
    }

    // Step 3: normalize = (x - mean) / sqrt(var + eps) * gamma + beta
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%in, %tmp_mean, %tmp_var, %gamma, %beta :
          memref<4x8xf32>, memref<4xf32>, memref<4xf32>,
          memref<8xf32>, memref<8xf32>)
      outs(%out : memref<4x8xf32>) {
    ^bb0(%x: f32, %mean: f32, %var: f32, %g: f32, %b: f32, %dummy: f32):
      %diff = arith.subf %x, %mean : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %normed = arith.mulf %diff, %inv_std : f32
      %scaled = arith.mulf %normed, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    }

    return
  }
}
