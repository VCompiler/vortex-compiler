// RUN: not vx-opt %s 2>&1 | FileCheck %s

func.func @bad(%pred: i1) {
  vortex.predicated %pred {
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: 'vortex.predicated' op must be nested inside vortex.launch
