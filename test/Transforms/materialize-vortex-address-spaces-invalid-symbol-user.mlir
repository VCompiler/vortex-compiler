// RUN: not vx-opt %s --vortex-mark-kernel='kernel-name=kernel' --vortex-materialize-address-spaces 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<4xf32>) attributes {vortex.kernel} {
  return
}

func.func @caller(%arg0: memref<4xf32>) {
  func.call @kernel(%arg0) : (memref<4xf32>) -> ()
  return
}

// CHECK: error:
// CHECK-SAME: cannot materialize Vortex address spaces on a kernel with symbol users yet
