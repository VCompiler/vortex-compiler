// RUN: not vx-opt %s --vortex-promote-tiles-to-local 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %tile = memref.subview %arg0[%c0] [8] [1] {vortex.promote_to_local} :
    memref<16xf32, #vortex.address_space<global>> to
    memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
  return
}

// CHECK: error:
// CHECK-SAME: requires enclosing vortex.launch for local promotion
