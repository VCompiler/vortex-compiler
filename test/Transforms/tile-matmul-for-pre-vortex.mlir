// RUN: vx-opt %s --allow-unregistered-dialect --vortex-tile-matmul-for-pre-vortex='tile-size=8' | FileCheck %s

func.func @main_graph(%arg0: memref<16x16xf32>,
                      %arg1: memref<16x16xf32>,
                      %arg2: memref<16x16xf32>) attributes {vortex.entry} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
  linalg.matmul ins(%arg0, %arg1 : memref<16x16xf32>, memref<16x16xf32>)
      outs(%alloc : memref<16x16xf32>)
  memref.copy %alloc, %arg2 : memref<16x16xf32> to memref<16x16xf32>
  return
}

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[C8:.+]] = arith.constant 8 : index
// CHECK: %[[C16:.+]] = arith.constant 16 : index
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[CTILE:.+]] = memref.subview %arg2
// CHECK: linalg.fill ins(%cst : f32) outs(%[[CTILE]]
// CHECK: scf.for
// CHECK: %[[ATILE:.+]] = memref.subview %arg0
// CHECK: %[[BTILE:.+]] = memref.subview %arg1
// CHECK: linalg.matmul ins(%[[ATILE]], %[[BTILE]]
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
