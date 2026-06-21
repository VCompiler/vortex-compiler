// RUN: vx-opt %s | FileCheck %s

func.func @simt_ops(%pred: i1) {
  %c1 = arith.constant 1 : index
  vortex.launch %c1, %c1, %c1 {
    vortex.predicated %pred {
      vortex.yield
    }
    vortex.divergent_if %pred {
      %mask = vortex.tmask : index
      vortex.tmc %mask
      vortex.pred %pred, %mask
      vortex.pred_n %pred, %mask
      %sp = vortex.split %pred : index
      %spn = vortex.split_n %pred : index
      vortex.join %sp
      vortex.join %spn
      vortex.yield
    } else {
      vortex.yield
    }
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @simt_ops
// CHECK: vortex.launch %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK: vortex.predicated %{{.*}} {
// CHECK: vortex.yield
// CHECK: vortex.divergent_if %{{.*}} {
// CHECK: %[[MASK:.*]] = vortex.tmask : index
// CHECK: vortex.tmc %[[MASK]]
// CHECK: vortex.pred %{{.*}}, %[[MASK]]
// CHECK: vortex.pred_n %{{.*}}, %[[MASK]]
// CHECK: %[[SP:.*]] = vortex.split %{{.*}} : index
// CHECK: %[[SPN:.*]] = vortex.split_n %{{.*}} : index
// CHECK: vortex.join %[[SP]]
// CHECK: vortex.join %[[SPN]]
// CHECK: vortex.yield
// CHECK: } else {
// CHECK: vortex.yield
