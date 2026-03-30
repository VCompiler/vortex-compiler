// RUN: not vx-opt %s --vortex-promote-tiles-to-local 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f32

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [8] [1] {vortex.promote_to_local} :
      memref<16xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    memref.store %cst, %tile[%c0] : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: requires vortex.write_back when promoted tile has write uses
