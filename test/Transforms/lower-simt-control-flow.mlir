// RUN: vx-opt %s --vortex-lower-simt-control-flow | FileCheck %s

func.func @lower_structured(%out: memref<16xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %v0 = arith.constant 0 : i32
  %v1 = arith.constant 1 : i32
  %v2 = arith.constant 2 : i32
  vortex.launch %c1, %c1, %c4 {
    %tid = vortex.thread_id : index
    %pred = arith.cmpi eq, %tid, %c0 : index
    vortex.predicated %pred {
      memref.store %v0, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
      vortex.yield
    }
    vortex.divergent_if %pred {
      memref.store %v1, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
      vortex.yield
    } else {
      memref.store %v2, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
      vortex.yield
    }
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @lower_structured
// CHECK: vortex.launch %{{.*}}, %{{.*}}, %[[THREADS:.*]] {
// CHECK: %[[RESTORE_MASK:.*]] = vortex.tmask : index
// CHECK: %[[THREADS_I64:.*]] = arith.index_cast %[[THREADS]] : index to i64
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[MASK_PLUS_ONE:.*]] = arith.shli %[[ONE]], %[[THREADS_I64]] : i64
// CHECK: %[[MASK_I64:.*]] = arith.subi %[[MASK_PLUS_ONE]], %[[ONE]] : i64
// CHECK: %[[MASK:.*]] = arith.index_cast %[[MASK_I64]] : i64 to index
// CHECK: vortex.tmc %[[MASK]]
// CHECK: %[[PRED:.*]] = arith.cmpi
// CHECK: %[[SP0:.*]] = vortex.split %[[PRED]] : index
// CHECK-NEXT: scf.if %[[PRED]] {
// CHECK: memref.store
// CHECK: }
// CHECK: vortex.join %[[SP0]]
// CHECK: %[[SP1:.*]] = vortex.split %[[PRED]] : index
// CHECK-NEXT: scf.if %[[PRED]] {
// CHECK: memref.store
// CHECK: } else {
// CHECK: memref.store
// CHECK: }
// CHECK: vortex.join %[[SP1]]
// CHECK: vortex.tmc %[[RESTORE_MASK]]
// CHECK-NOT: vortex.predicated
// CHECK-NOT: vortex.divergent_if
// CHECK-NOT: vortex.pred
