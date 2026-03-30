// RUN: not vx-opt %s 2>&1 | FileCheck %s

module attributes {vortex.kernel} {
}

// CHECK: error:
// CHECK-SAME: 'vortex.kernel' may only annotate func.func
