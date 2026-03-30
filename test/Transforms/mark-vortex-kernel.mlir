// RUN: vx-opt %s --vortex-mark-kernel='kernel-name=tiled_matmul remove-entry-attr=true' | FileCheck %s

func.func @helper(%arg0: memref<4xf32>) {
  return
}

func.func @tiled_matmul(%arg0: memref<4xf32>) {
  return
}

func.func @temporary_entry(%arg0: memref<4xf32>) attributes {vortex.entry} {
  return
}

// CHECK-LABEL: func.func @helper(
// CHECK-NOT: vortex.kernel

// CHECK-LABEL: func.func @tiled_matmul(
// CHECK-SAME: attributes {vortex.kernel}

// CHECK-LABEL: func.func @temporary_entry(
// CHECK-SAME: attributes {vortex.kernel}
// CHECK-NOT: vortex.entry
