// RUN: not vx-opt %s --vortex-promote-tiles-to-local 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<?xf32, #vortex.address_space<global>>, %n: index) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [%n] [1] {vortex.promote_to_local} :
      memref<?xf32, #vortex.address_space<global>> to
      memref<?xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: currently only static-shaped tiles can be promoted to local memory

