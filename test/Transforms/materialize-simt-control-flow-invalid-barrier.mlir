// RUN: not vx-opt %s --vortex-materialize-simt-control-flow 2>&1 | FileCheck %s

func.func @bad_barrier() attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  vortex.launch %c1, %c1, %c1 {
    %tid = vortex.thread_id : index
    %pred = arith.cmpi eq, %tid, %c0 : index
    scf.if %pred {
      vortex.barrier <core>
    }
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: cannot materialize may-varying scf.if containing vortex.barrier
