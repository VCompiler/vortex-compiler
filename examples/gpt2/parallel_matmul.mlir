// Parallel matmul: C[4,8] = A[4,8] @ B[8,8]
// M dimension (4 rows) mapped to 4 cores
module {
  func.func @parallel_matmul(%a: memref<4x8xf32>, %b: memref<8x8xf32>,
                             %c: memref<4x8xf32>)
      attributes {vortex.entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32

    // M dimension mapped to cores: each core computes one row
    // Each core zeroes and accumulates its own row — no cross-core data race
    "scf.for"(%c0, %c4, %c1) ({
    ^bb0(%i: index):
      // Zero this row
      scf.for %jz = %c0 to %c8 step %c1 {
        memref.store %zero, %c[%i, %jz] : memref<4x8xf32>
      }
      // Compute this row
      scf.for %j = %c0 to %c8 step %c1 {
        scf.for %k = %c0 to %c8 step %c1 {
          %a_val = memref.load %a[%i, %k] : memref<4x8xf32>
          %b_val = memref.load %b[%k, %j] : memref<8x8xf32>
          %c_val = memref.load %c[%i, %j] : memref<4x8xf32>
          %prod = arith.mulf %a_val, %b_val : f32
          %sum = arith.addf %c_val, %prod : f32
          memref.store %sum, %c[%i, %j] : memref<4x8xf32>
        }
      }
      scf.yield
    }) {vortex.mapping = "core"} : (index, index, index) -> ()

    return
  }
}
