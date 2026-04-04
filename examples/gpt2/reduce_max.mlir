// reduce_max: find max element of a 1-D f32 buffer.
// input:  memref<16xf32>
// output: memref<1xf32>
module {
  func.func @reduce_max(%in: memref<16xf32>, %out: memref<1xf32>)
      attributes {vortex.entry} {
    %neg_inf = arith.constant 0xFF800000 : f32  // -inf
    linalg.fill ins(%neg_inf : f32) outs(%out : memref<1xf32>)

    linalg.generic {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (0)>
      ],
      iterator_types = ["reduction"]
    } ins(%in : memref<16xf32>)
      outs(%out : memref<1xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %mx = arith.maximumf %a, %acc : f32
      linalg.yield %mx : f32
    }
    return
  }
}
