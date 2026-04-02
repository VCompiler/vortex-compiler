// RUN: not vx-opt %s --vortex-plan-local-memory-layout 2>&1 | FileCheck %s

func.func @kernel(%n: index) attributes {vortex.kernel} {
  %buf = vortex.local_alloc(%n) : memref<?xf32, #vortex.address_space<local>>
  func.call @touch(%buf) : (memref<?xf32, #vortex.address_space<local>>) -> ()
  return
}

func.func private @touch(%arg0: memref<?xf32, #vortex.address_space<local>>)

// CHECK: error:
// CHECK-SAME: currently requires static-shaped vortex.local_alloc
