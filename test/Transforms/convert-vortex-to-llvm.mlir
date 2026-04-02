// RUN: vx-opt %s --vortex-lower-runtime-builtins | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  %cid = vortex.core_id : index
  %wid = vortex.subgroup_id : index
  %tid = vortex.thread_id : index
  %sum0 = arith.addi %cid, %wid : index
  %sum1 = arith.addi %sum0, %tid : index
  vortex.barrier <core>
  %sum2 = arith.addi %sum1, %cid : index
  %sum3 = arith.addi %sum2, %wid : index
  %sum4 = arith.addi %sum3, %tid : index
  return
}

// CHECK-DAG: func.func private @vx_core_id() -> i32
// CHECK-DAG: func.func private @vx_warp_id() -> i32
// CHECK-DAG: func.func private @vx_thread_id() -> i32
// CHECK-DAG: func.func private @vx_num_warps() -> i32
// CHECK-DAG: func.func private @vx_barrier(i32, i32)
// CHECK-LABEL: func.func @kernel() attributes {vortex.kernel_entry} {
// CHECK: %[[CID_I32:.*]] = call @vx_core_id() : () -> i32
// CHECK: %[[CID:.*]] = arith.index_cast %[[CID_I32]] : i32 to index
// CHECK: %[[WID_I32:.*]] = call @vx_warp_id() : () -> i32
// CHECK: %[[WID:.*]] = arith.index_cast %[[WID_I32]] : i32 to index
// CHECK: %[[TID_I32:.*]] = call @vx_thread_id() : () -> i32
// CHECK: %[[TID:.*]] = arith.index_cast %[[TID_I32]] : i32 to index
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[NW:.*]] = call @vx_num_warps() : () -> i32
// CHECK: call @vx_barrier(%[[ZERO]], %[[NW]]) : (i32, i32) -> ()
// CHECK-NOT: vortex.core_id
// CHECK-NOT: vortex.subgroup_id
// CHECK-NOT: vortex.thread_id
// CHECK-NOT: vortex.barrier
// CHECK-NOT: vortex.kernel
