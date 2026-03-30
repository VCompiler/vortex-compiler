// RUN: not vx-opt %s 2>&1 | FileCheck %s

func.func @bad() {
  %buf = vortex.local_alloc() : memref<4xf32, #vortex.address_space<private>>
  return
}

// CHECK: error:
// CHECK-SAME: result memref must use #vortex.address_space<local>
