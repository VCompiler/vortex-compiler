// RUN: vx-opt %s --allow-unregistered-dialect --vortex-normalize-onnx-frontend | FileCheck %s

module attributes {"onnx-mlir.symbol-postfix" = "matmul_16x16"} {
  func.func @main_graph(%arg0: memref<16x16xf32> {onnx.name = "A"},
                        %arg1: memref<16x16xf32> {onnx.name = "B"})
      -> (memref<16x16xf32> {onnx.name = "C"}) attributes {onnx.test = "x"} {
    return %arg0 : memref<16x16xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// CHECK-LABEL: module
// CHECK: func.func @main_graph(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> attributes {vortex.entry}
// CHECK-NOT: onnx.name
// CHECK-NOT: onnx-mlir.symbol-postfix
// CHECK-NOT: "onnx.EntryPoint"
