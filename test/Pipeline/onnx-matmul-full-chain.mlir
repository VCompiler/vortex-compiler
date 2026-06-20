// RUN: vx-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(vortex-onnx-matmul-to-pre-vortex-pipeline{tile-size=2},func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces,vortex-map-parallel-loops-to-launch,vortex-promote-tiles-to-local,vortex-insert-barriers,vortex-distribute-local-copies,vortex-plan-local-memory-layout),vortex-lower-local-memory,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse)' | FileCheck %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
                   llvm.target_triple = "x86_64-unknown-linux-gnu",
                   "onnx-mlir.symbol-postfix" = "matmul_4x4"} {
  func.func @main_graph(%arg0: memref<4x4xf32> {onnx.name = "A"},
                        %arg1: memref<4x4xf32> {onnx.name = "B"})
      -> (memref<4x4xf32> {onnx.name = "C"}) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<4x4xf32>)
    linalg.matmul ins(%arg0, %arg1 : memref<4x4xf32>, memref<4x4xf32>)
        outs(%alloc : memref<4x4xf32>)
    return %alloc : memref<4x4xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// CHECK-DAG: func.func private @vx_local_mem_base() -> i64
// CHECK-DAG: func.func private @vx_barrier(i32, i32)
// CHECK-DAG: func.func private @vx_warp_id() -> i32
// CHECK-DAG: func.func private @vx_thread_id() -> i32
// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME: attributes {vortex.kernel_entry
// CHECK-SAME: vortex.local_frame_bytes = 32 : i64
// CHECK: call @vx_local_mem_base
// CHECK: func.call @vx_warp_id
// CHECK: func.call @vx_thread_id
// CHECK: llvm.store
// CHECK: func.call @vx_barrier
// CHECK: llvm.load
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: "onnx.EntryPoint"
// CHECK-NOT: linalg.matmul
// CHECK-NOT: vortex.launch
// CHECK-NOT: vortex.local_alloc
// CHECK-NOT: vortex.thread_id
