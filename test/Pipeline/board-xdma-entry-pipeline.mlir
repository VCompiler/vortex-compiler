// RUN: vx-opt %s --pass-pipeline='builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,vortex-materialize-board-xdma-entry,reconcile-unrealized-casts)' | FileCheck %s

module {
  func.func @store_one(%out: memref<4xf32>) attributes {vortex.entry} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.000000e+00 : f32
    memref.store %one, %out[%c0] : memref<4xf32>
    return
  }
}

// CHECK-DAG: llvm.func @vortex_board_xdma_startup_arg() -> i32 attributes {sym_visibility = "private"}
// CHECK-DAG: llvm.func @vortex_board_xdma_exit(i32) attributes {sym_visibility = "private"}
// CHECK-LABEL: llvm.func @store_one(%arg0: !llvm.ptr) attributes {vortex.kernel_entry}
// CHECK-LABEL: llvm.func @main() -> i32
// CHECK: llvm.call @vortex_board_xdma_startup_arg() : () -> i32
// CHECK: llvm.getelementptr {{.*}}[8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: llvm.load {{.*}} {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK: llvm.call @store_one({{.*}}) : (!llvm.ptr) -> ()
// CHECK: llvm.call @vortex_board_xdma_exit
