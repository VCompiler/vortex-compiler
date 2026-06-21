// RUN: vx-opt %s --vortex-lower-runtime-builtins | FileCheck %s

func.func @simt_builtins(%pred: i1, %restore: index) attributes {vortex.kernel} {
  %mask = vortex.tmask : index
  vortex.tmc %mask
  vortex.pred %pred, %mask
  vortex.pred_n %pred, %restore
  %sp = vortex.split %pred : index
  %spn = vortex.split_n %pred : index
  vortex.join %sp
  vortex.join %spn
  return
}

// CHECK-DAG: func.func private @vx_thread_mask() -> i32
// CHECK-DAG: func.func private @vx_tmc(i32)
// CHECK-DAG: func.func private @vx_pred(i32, i32)
// CHECK-DAG: func.func private @vx_pred_n(i32, i32)
// CHECK-DAG: func.func private @vx_split(i32) -> i32
// CHECK-DAG: func.func private @vx_split_n(i32) -> i32
// CHECK-DAG: func.func private @vx_join(i32)
// CHECK-LABEL: func.func @simt_builtins(
// CHECK-SAME: %[[PRED:.*]]: i1, %[[RESTORE:.*]]: index
// CHECK-SAME: attributes {vortex.kernel_entry} {
// CHECK: %[[MASK_I32:.*]] = call @vx_thread_mask() : () -> i32
// CHECK: %[[MASK:.*]] = arith.index_cast %[[MASK_I32]] : i32 to index
// CHECK: %[[TMC_MASK:.*]] = arith.index_cast %[[MASK]] : index to i32
// CHECK: call @vx_tmc(%[[TMC_MASK]]) : (i32) -> ()
// CHECK: %[[PRED_I32:.*]] = arith.extui %[[PRED]] : i1 to i32
// CHECK: %[[MASK_FOR_PRED:.*]] = arith.index_cast %[[MASK]] : index to i32
// CHECK: call @vx_pred(%[[PRED_I32]], %[[MASK_FOR_PRED]]) : (i32, i32) -> ()
// CHECK: %[[PRED_N_I32:.*]] = arith.extui %[[PRED]] : i1 to i32
// CHECK: %[[RESTORE_I32:.*]] = arith.index_cast %[[RESTORE]] : index to i32
// CHECK: call @vx_pred_n(%[[PRED_N_I32]], %[[RESTORE_I32]]) : (i32, i32) -> ()
// CHECK: %[[SPLIT_PRED_I32:.*]] = arith.extui %[[PRED]] : i1 to i32
// CHECK: %[[SP_I32:.*]] = call @vx_split(%[[SPLIT_PRED_I32]]) : (i32) -> i32
// CHECK: %[[SP:.*]] = arith.index_cast %[[SP_I32]] : i32 to index
// CHECK: %[[SPLIT_N_PRED_I32:.*]] = arith.extui %[[PRED]] : i1 to i32
// CHECK: %[[SPN_I32:.*]] = call @vx_split_n(%[[SPLIT_N_PRED_I32]]) : (i32) -> i32
// CHECK: %[[SPN:.*]] = arith.index_cast %[[SPN_I32]] : i32 to index
// CHECK: %[[JOIN_SP_I32:.*]] = arith.index_cast %[[SP]] : index to i32
// CHECK: call @vx_join(%[[JOIN_SP_I32]]) : (i32) -> ()
// CHECK: %[[JOIN_SPN_I32:.*]] = arith.index_cast %[[SPN]] : index to i32
// CHECK: call @vx_join(%[[JOIN_SPN_I32]]) : (i32) -> ()
// CHECK-NOT: vortex.tmask
// CHECK-NOT: vortex.tmc
// CHECK-NOT: vortex.pred
// CHECK-NOT: vortex.split
// CHECK-NOT: vortex.join
