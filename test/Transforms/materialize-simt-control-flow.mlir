// RUN: vx-opt %s --vortex-materialize-simt-control-flow | FileCheck %s

func.func @predicated(%out: memref<16xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %v0 = arith.constant 0 : i32
  vortex.launch %c1, %c1, %c1 {
    %tid = vortex.thread_id : index
    %pred = arith.cmpi eq, %tid, %c0 : index
    scf.if %pred {
      memref.store %v0, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
    }
    vortex.yield
  }
  return
}

func.func @uniform_conditions(
    %out: memref<16xi32, #vortex.address_space<global>>, %arg_pred: i1)
    attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %v0 = arith.constant 0 : i32
  %true = arith.constant true
  vortex.launch %c1, %c1, %c1 {
    %core = vortex.core_id : index
    %core_pred = arith.cmpi eq, %core, %c0 : index
    scf.if %core_pred {
      memref.store %v0, %out[%c0] : memref<16xi32, #vortex.address_space<global>>
    }
    %subgroup = vortex.subgroup_id : index
    %subgroup_pred = arith.cmpi eq, %subgroup, %c0 : index
    scf.if %subgroup_pred {
      memref.store %v0, %out[%c0] : memref<16xi32, #vortex.address_space<global>>
    }
    scf.if %true {
      memref.store %v0, %out[%c0] : memref<16xi32, #vortex.address_space<global>>
    }
    scf.if %arg_pred {
      memref.store %v0, %out[%c0] : memref<16xi32, #vortex.address_space<global>>
    }
    vortex.yield
  }
  return
}

func.func @divergent_else(%out: memref<16xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %v0 = arith.constant 0 : i32
  %v1 = arith.constant 1 : i32
  vortex.launch %c1, %c1, %c1 {
    %tid = vortex.thread_id : index
    %pred = arith.cmpi eq, %tid, %c0 : index
    scf.if %pred {
      memref.store %v0, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
    } else {
      memref.store %v1, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
    }
    vortex.yield
  }
  return
}


func.func private @opaque_cond() -> i1

func.func @call_condition_may_varying(%out: memref<16xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %v0 = arith.constant 0 : i32
  vortex.launch %c1, %c1, %c1 {
    %pred = func.call @opaque_cond() : () -> i1
    scf.if %pred {
      memref.store %v0, %out[%c0] : memref<16xi32, #vortex.address_space<global>>
    }
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @predicated
// CHECK: vortex.predicated %{{.*}} {
// CHECK: memref.store
// CHECK: vortex.yield
// CHECK-NOT: scf.if

// CHECK-LABEL: func.func @uniform_conditions
// CHECK: arith.constant true
// CHECK: vortex.core_id
// CHECK: scf.if
// CHECK: vortex.subgroup_id
// CHECK: scf.if
// CHECK: scf.if
// CHECK: scf.if
// CHECK-NOT: vortex.predicated
// CHECK-NOT: vortex.divergent_if

// CHECK-LABEL: func.func @divergent_else
// CHECK: vortex.divergent_if %{{.*}} {
// CHECK: memref.store
// CHECK: vortex.yield
// CHECK: } else {
// CHECK: memref.store
// CHECK: vortex.yield
// CHECK-NOT: scf.if

// CHECK-LABEL: func.func @call_condition_may_varying
// CHECK: %[[CALL:.*]] = func.call @opaque_cond
// CHECK: vortex.predicated %[[CALL]] {
// CHECK: memref.store
// CHECK: vortex.yield
// CHECK-NOT: scf.if
