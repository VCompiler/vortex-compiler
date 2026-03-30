module {
  func.func @tiled_matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) attributes {vortex.pre_vortex_dialects = ["arith", "linalg", "memref", "scf"], vortex.pre_vortex_memory_spaces = ["<default>"], vortex.pre_vortex_ops = ["arith.addf", "arith.mulf", "linalg.fill", "linalg.matmul", "memref.subview", "scf.for"]} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.for %arg3 = %c0 to %c128 step %c8 {
      scf.for %arg4 = %c0 to %c128 step %c8 {
        %subview = memref.subview %arg2[%arg3, %arg4] [8, 8] [1, 1] : memref<128x128xf32> to memref<8x8xf32, strided<[128, 1], offset: ?>>
        linalg.fill ins(%cst : f32) outs(%subview : memref<8x8xf32, strided<[128, 1], offset: ?>>)
        scf.for %arg5 = %c0 to %c128 step %c8 {
          %subview_0 = memref.subview %arg0[%arg3, %arg5] [8, 8] [1, 1] : memref<128x128xf32> to memref<8x8xf32, strided<[128, 1], offset: ?>>
          %subview_1 = memref.subview %arg1[%arg5, %arg4] [8, 8] [1, 1] : memref<128x128xf32> to memref<8x8xf32, strided<[128, 1], offset: ?>>
          linalg.matmul ins(%subview_0, %subview_1 : memref<8x8xf32, strided<[128, 1], offset: ?>>, memref<8x8xf32, strided<[128, 1], offset: ?>>) outs(%subview : memref<8x8xf32, strided<[128, 1], offset: ?>>)
        }
      }
    }
    return
  }
}

