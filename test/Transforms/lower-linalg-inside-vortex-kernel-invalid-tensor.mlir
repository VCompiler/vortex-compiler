// RUN: not vx-opt %s --vortex-lower-linalg-inside-kernel 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32

  vortex.launch %c1, %c1, %c1 {
    %empty = tensor.empty() : tensor<4xf32>
    %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    vortex.yield
  }
  return
}

// CHECK: error: 'linalg.fill' op requires buffer semantics before vortex-lower-linalg-inside-kernel
