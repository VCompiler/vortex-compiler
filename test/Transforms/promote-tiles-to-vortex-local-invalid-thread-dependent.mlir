// RUN: not vx-opt %s --vortex-promote-tiles-to-local 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<32xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %tid = vortex.thread_id : index
    %tile = memref.subview %arg0[%tid] [8] [1] {vortex.promote_to_local} :
      memref<32xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: requires promoted tiles to stay uniform across subgroup/thread

