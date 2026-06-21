// RUN: not vx-opt %s 2>&1 | FileCheck %s

func.func @bad(%pred: i1) {
  vortex.divergent_if %pred {
    vortex.yield
  } else {
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: 'vortex.divergent_if' op must be nested inside vortex.launch
