// RUN: not vx-opt %s --vortex-plan-local-memory-layout 2>&1 | FileCheck %s

func.func private @use_memref(%value: memref<?xf32, #vortex.address_space<local>>)

func.func @kernel() attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %buf = vortex.local_alloc() : memref<4xf32, #vortex.address_space<local>>
    %cast = memref.cast %buf : memref<4xf32, #vortex.address_space<local>> to memref<?xf32, #vortex.address_space<local>>
    func.call @use_memref(%cast) : (memref<?xf32, #vortex.address_space<local>>) -> ()
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: cannot escape local memory planning scope through call
