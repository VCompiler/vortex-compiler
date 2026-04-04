// reduce_sum: sum all elements of a 1-D f32 buffer into a scalar output.
// input:  memref<16xf32>
// output: memref<1xf32>  (holds the scalar sum)
module {
  func.func @reduce_sum(%in: memref<16xf32>, %out: memref<1xf32>)
      attributes {vortex.entry} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%out : memref<1xf32>)

    %c0 = arith.constant 0 : index
    linalg.generic {
      indexing_maps = [
        affine_map<(i) -> (i)>,    // in
        affine_map<(i) -> (0)>     // out (reduction target)
      ],
      iterator_types = ["reduction"]
    } ins(%in : memref<16xf32>)
      outs(%out : memref<1xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %sum = arith.addf %a, %acc : f32
      linalg.yield %sum : f32
    }
    return
  }
}
