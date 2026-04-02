// RUN: vx-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(vortex-onnx-matmul-to-pre-vortex-pipeline)' | FileCheck %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
                   llvm.target_triple = "x86_64-unknown-linux-gnu",
                   "onnx-mlir.symbol-postfix" = "matmul_16x16"} {
  func.func @main_graph(%arg0: memref<16x16xf32> {onnx.name = "A"},
                        %arg1: memref<16x16xf32> {onnx.name = "B"})
      -> (memref<16x16xf32> {onnx.name = "C"}) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
    linalg.matmul ins(%arg0, %arg1 : memref<16x16xf32>, memref<16x16xf32>)
        outs(%alloc : memref<16x16xf32>)
    return %alloc : memref<16x16xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// CHECK-LABEL: func.func @main_graph(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) attributes
// CHECK-SAME: vortex.entry
// CHECK-SAME: vortex.pre_vortex_dialects = ["arith", "linalg", "memref", "scf"]
// CHECK: %[[CTILE:.+]] = memref.subview %arg2
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK-NOT: "onnx.EntryPoint"
