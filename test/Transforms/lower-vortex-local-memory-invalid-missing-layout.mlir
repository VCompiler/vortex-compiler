// RUN: not vx-opt %s --vortex-lower-local-memory 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  %buf = vortex.local_alloc() : memref<4xf32, #vortex.address_space<local>>
  return
}

// CHECK: error:
// CHECK-SAME: requires running vortex-plan-local-memory-layout first
