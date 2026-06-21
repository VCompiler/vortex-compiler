// RUN: vx-opt %s --vortex-materialize-simt-control-flow --vortex-lower-simt-control-flow | FileCheck %s

func.func @materialize_then_lower(%out: memref<16xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %v0 = arith.constant 0 : i32
  %v1 = arith.constant 1 : i32
  vortex.launch %c1, %c1, %c4 {
    %tid = vortex.thread_id : index
    %pred = arith.cmpi eq, %tid, %c0 : index
    scf.if %pred {
      memref.store %v0, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
    }
    scf.if %pred {
      memref.store %v0, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
    } else {
      memref.store %v1, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
    }
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @materialize_then_lower
// CHECK: vortex.tmc %{{.*}}
// CHECK: %[[SP0:.*]] = vortex.split %[[PRED:.*]] : index
// CHECK-NEXT: scf.if %[[PRED]] {
// CHECK: memref.store
// CHECK: vortex.join %[[SP0]]
// CHECK: %[[SP1:.*]] = vortex.split %[[PRED]] : index
// CHECK-NEXT: scf.if %[[PRED]] {
// CHECK: memref.store
// CHECK: } else {
// CHECK: memref.store
// CHECK: vortex.join %[[SP1]]
// CHECK: vortex.tmc %{{.*}}
// CHECK-NOT: vortex.predicated
// CHECK-NOT: vortex.divergent_if
