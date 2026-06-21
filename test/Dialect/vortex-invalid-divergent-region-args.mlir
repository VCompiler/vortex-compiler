// RUN: not vx-opt %s 2>&1 | FileCheck %s

func.func @bad(%pred: i1) {
  %c1 = arith.constant 1 : index
  vortex.launch %c1, %c1, %c1 {
    vortex.divergent_if %pred {
    ^bb0(%illegal: index):
      vortex.yield
    } else {
      vortex.yield
    }
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: expects divergent_if then region without block arguments
