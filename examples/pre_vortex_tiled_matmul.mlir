module {
  func.func @tiled_matmul(
      %a: memref<128x128xf32>,
      %b: memref<128x128xf32>,
      %c: memref<128x128xf32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.0 : f32

    scf.for %ii = %c0 to %c128 step %c8 {
      scf.for %jj = %c0 to %c128 step %c8 {
        %c_tile = memref.subview %c[%ii, %jj] [8, 8] [1, 1]
          : memref<128x128xf32> to memref<8x8xf32, strided<[128, 1], offset: ?>>
        linalg.fill ins(%zero : f32)
            outs(%c_tile : memref<8x8xf32, strided<[128, 1], offset: ?>>)

        scf.for %kk = %c0 to %c128 step %c8 {
          %a_tile = memref.subview %a[%ii, %kk] [8, 8] [1, 1]
            : memref<128x128xf32> to memref<8x8xf32, strided<[128, 1], offset: ?>>
          %b_tile = memref.subview %b[%kk, %jj] [8, 8] [1, 1]
            : memref<128x128xf32> to memref<8x8xf32, strided<[128, 1], offset: ?>>

          linalg.matmul
              ins(%a_tile, %b_tile :
                    memref<8x8xf32, strided<[128, 1], offset: ?>>,
                    memref<8x8xf32, strided<[128, 1], offset: ?>>)
              outs(%c_tile :
                    memref<8x8xf32, strided<[128, 1], offset: ?>>)
        }
      }
    }
    return
  }
}
