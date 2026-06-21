// RUN: not vx-opt %s --vortex-materialize-simt-control-flow 2>&1 | FileCheck %s

func.func @bad_nested(%out: memref<16xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %v0 = arith.constant 0 : i32
  %true = arith.constant true
  vortex.launch %c1, %c1, %c1 {
    %tid = vortex.thread_id : index
    %pred = arith.cmpi eq, %tid, %c0 : index
    scf.if %pred {
      scf.if %true {
        memref.store %v0, %out[%tid] : memref<16xi32, #vortex.address_space<global>>
      }
    }
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: cannot materialize may-varying scf.if with nested complex control flow
