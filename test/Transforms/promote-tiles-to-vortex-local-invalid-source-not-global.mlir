// RUN: not vx-opt %s --vortex-promote-tiles-to-local 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<16xf32>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [8] [1] {vortex.promote_to_local} :
      memref<16xf32> to memref<8xf32, strided<[1], offset: ?>>
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: requires source/base memref to use explicit #vortex.address_space<global>

