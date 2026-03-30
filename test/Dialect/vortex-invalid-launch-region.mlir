// RUN: not vx-opt %s 2>&1 | FileCheck %s

func.func @bad(%n: index) {
  vortex.launch %n, %n, %n {
  ^bb0(%illegal: index):
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: expects launch body without block arguments
