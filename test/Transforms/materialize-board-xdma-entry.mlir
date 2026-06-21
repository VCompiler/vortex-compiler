// RUN: vx-opt %s --vortex-materialize-board-xdma-entry | FileCheck %s

module {
  llvm.func @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
                    %arg2: !llvm.ptr) attributes {vortex.kernel_entry} {
    llvm.return
  }
}

// CHECK-DAG: llvm.func @vortex_board_xdma_startup_arg() -> i32 attributes {sym_visibility = "private"}
// CHECK-DAG: llvm.func @vortex_board_xdma_exit(i32) attributes {sym_visibility = "private"}
// CHECK-DAG: llvm.func @vx_tmc(i32) attributes {sym_visibility = "private"}
// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: attributes {vortex.kernel_entry}
// CHECK-LABEL: llvm.func @main() -> i32 {
// CHECK: %[[CONTROL_MASK:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @vx_tmc(%[[CONTROL_MASK]]) : (i32) -> ()
// CHECK: %[[DESC_RAW:.*]] = llvm.call @vortex_board_xdma_startup_arg() : () -> i32
// CHECK: %[[DESC:.*]] = llvm.inttoptr %[[DESC_RAW]] : i32 to !llvm.ptr
// CHECK: %[[SLOT0:.*]] = llvm.getelementptr %[[DESC]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[ADDR0:.*]] = llvm.load %[[SLOT0]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK: %[[ARG0:.*]] = llvm.inttoptr %[[ADDR0]] : i64 to !llvm.ptr
// CHECK: %[[SLOT1:.*]] = llvm.getelementptr %[[DESC]][16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[ADDR1:.*]] = llvm.load %[[SLOT1]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK: %[[ARG1:.*]] = llvm.inttoptr %[[ADDR1]] : i64 to !llvm.ptr
// CHECK: %[[SLOT2:.*]] = llvm.getelementptr %[[DESC]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[ADDR2:.*]] = llvm.load %[[SLOT2]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK: %[[ARG2:.*]] = llvm.inttoptr %[[ADDR2]] : i64 to !llvm.ptr
// CHECK: llvm.call @kernel(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK: %[[STATUS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @vortex_board_xdma_exit(%[[STATUS]]) : (i32) -> ()
// CHECK: llvm.return {{.*}} : i32
