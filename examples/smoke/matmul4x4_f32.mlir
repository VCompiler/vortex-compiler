module {
  func.func @matmul4x4(%a: memref<4x4xf32>, %b: memref<4x4xf32>,
                       %c: memref<4x4xf32>) attributes {vortex.entry} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<4x4xf32>)
    linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
        outs(%c : memref<4x4xf32>)
    return
  }
}
