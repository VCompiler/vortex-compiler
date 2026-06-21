// RUN: vx-opt %s --vortex-materialize-board-xdma-entry | FileCheck %s

module {
  llvm.func @vx_thread_mask() -> i32 attributes {sym_visibility = "private"}
  llvm.func @vx_tmc(i32) attributes {sym_visibility = "private"}

  llvm.func @kernel_vortex_launch_body(%arg0: !llvm.ptr) attributes {sym_visibility = "private"} {
    llvm.return
  }

  llvm.func @kernel(%arg0: !llvm.ptr) attributes {vortex.kernel_entry} {
    %mask = llvm.mlir.constant(15 : i32) : i32
    %restore = llvm.call @vx_thread_mask() : () -> i32
    llvm.call @vx_tmc(%mask) : (i32) -> ()
    llvm.call @kernel_vortex_launch_body(%arg0) : (!llvm.ptr) -> ()
    llvm.call @vx_tmc(%restore) : (i32) -> ()
    llvm.return
  }
}

// CHECK-DAG: llvm.func @vortex_board_xdma_spawn_threads_1d(i32, !llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK-DAG: llvm.func @vortex_board_xdma_startup_arg() -> i32 attributes {sym_visibility = "private"}
// CHECK-DAG: llvm.func @vortex_board_xdma_exit(i32) attributes {sym_visibility = "private"}
// CHECK-LABEL: llvm.func @main() -> i32 {
// CHECK: %[[DESC_RAW:.*]] = llvm.call @vortex_board_xdma_startup_arg() : () -> i32
// CHECK: %[[DESC:.*]] = llvm.inttoptr %[[DESC_RAW]] : i32 to !llvm.ptr
// CHECK: %[[THREADS:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: %[[ADAPTER:.*]] = llvm.mlir.addressof @kernel_vortex_launch_body_xdma_spawn_adapter : !llvm.ptr
// CHECK: llvm.call @vortex_board_xdma_spawn_threads_1d(%[[THREADS]], %[[ADAPTER]], %[[DESC]]) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @vortex_board_xdma_exit
// CHECK-LABEL: llvm.func @kernel_vortex_launch_body_xdma_spawn_adapter
// CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) attributes {sym_visibility = "private"}
// CHECK: %[[SLOT0:.*]] = llvm.getelementptr %[[ARG]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[ADDR0:.*]] = llvm.load %[[SLOT0]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK: %[[PTR0:.*]] = llvm.inttoptr %[[ADDR0]] : i64 to !llvm.ptr
// CHECK: llvm.call @kernel_vortex_launch_body(%[[PTR0]]) : (!llvm.ptr) -> ()
// CHECK: llvm.return
