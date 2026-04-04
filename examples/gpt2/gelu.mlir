// GeLU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// Elementwise on memref<16xf32>
module {
  func.func @gelu(%in: memref<16xf32>, %out: memref<16xf32>)
      attributes {vortex.entry} {
    %half = arith.constant 0.5 : f32
    %one = arith.constant 1.0 : f32
    %inv_sqrt2 = arith.constant 0.70710678118 : f32  // 1/sqrt(2)

    linalg.generic {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>
      ],
      iterator_types = ["parallel"]
    } ins(%in : memref<16xf32>)
      outs(%out : memref<16xf32>) {
    ^bb0(%x: f32, %dummy: f32):
      %x_scaled = arith.mulf %x, %inv_sqrt2 : f32
      %erf_val = math.erf %x_scaled : f32
      %one_plus_erf = arith.addf %one, %erf_val : f32
      %x_half = arith.mulf %x, %half : f32
      %result = arith.mulf %x_half, %one_plus_erf : f32
      linalg.yield %result : f32
    }
    return
  }
}
