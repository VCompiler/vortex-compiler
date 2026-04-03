// Numerically stable softmax on memref<4x8xf32> along axis=-1
// Steps: reduce_max → sub → exp → reduce_sum → div
module {
  func.func @softmax(%in: memref<4x8xf32>, %out: memref<4x8xf32>,
                     %tmp_max: memref<4xf32>, %tmp_sum: memref<4xf32>)
      attributes {vortex.entry} {

    // Step 1: reduce_max along axis=1
    %neg_inf = arith.constant 0xFF800000 : f32
    linalg.fill ins(%neg_inf : f32) outs(%tmp_max : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%in : memref<4x8xf32>) outs(%tmp_max : memref<4xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %mx = arith.maximumf %a, %acc : f32
      linalg.yield %mx : f32
    }

    // Step 2: exp(x - max) → out
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%in, %tmp_max : memref<4x8xf32>, memref<4xf32>)
      outs(%out : memref<4x8xf32>) {
    ^bb0(%x: f32, %mx: f32, %dummy: f32):
      %shifted = arith.subf %x, %mx : f32
      %e = math.exp %shifted : f32
      linalg.yield %e : f32
    }

    // Step 3: reduce_sum of exp values along axis=1
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%tmp_sum : memref<4xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%out : memref<4x8xf32>) outs(%tmp_sum : memref<4xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %s = arith.addf %a, %acc : f32
      linalg.yield %s : f32
    }

    // Step 4: divide by sum
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%out, %tmp_sum : memref<4x8xf32>, memref<4xf32>)
      outs(%out : memref<4x8xf32>) {
    ^bb0(%e: f32, %s: f32, %dummy: f32):
      %r = arith.divf %e, %s : f32
      linalg.yield %r : f32
    }

    return
  }
}
