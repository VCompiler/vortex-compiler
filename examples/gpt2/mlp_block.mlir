// MLP block: input @ W1 → GeLU → @ W2
// input:  memref<4x8xf32>   (seq_len=4, d_model=8)
// W1:     memref<8x32xf32>  (d_model -> 4*d_model)
// W2:     memref<32x8xf32>  (4*d_model -> d_model)
// hidden: memref<4x32xf32>  (scratch)
// output: memref<4x8xf32>
module {
  func.func @mlp_block(%in: memref<4x8xf32>, %w1: memref<8x32xf32>,
                       %w2: memref<32x8xf32>, %hidden: memref<4x32xf32>,
                       %out: memref<4x8xf32>)
      attributes {vortex.entry} {

    %zero = arith.constant 0.0 : f32
    %half = arith.constant 0.5 : f32
    %inv_sqrt2 = arith.constant 0.70710678118 : f32
    %one = arith.constant 1.0 : f32

    // Step 1: hidden = input @ W1
    linalg.fill ins(%zero : f32) outs(%hidden : memref<4x32xf32>)
    linalg.matmul ins(%in, %w1 : memref<4x8xf32>, memref<8x32xf32>)
        outs(%hidden : memref<4x32xf32>)

    // Step 2: hidden = GeLU(hidden)
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

    // Step 3: out = hidden @ W2
    linalg.fill ins(%zero : f32) outs(%out : memref<4x8xf32>)
    linalg.matmul ins(%hidden, %w2 : memref<4x32xf32>, memref<32x8xf32>)
        outs(%out : memref<4x8xf32>)

    return
  }
}
