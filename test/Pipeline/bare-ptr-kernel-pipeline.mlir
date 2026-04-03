// RUN: vx-opt %s --pass-pipeline='builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces,vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)' | FileCheck %s

module {
  func.func @matmul4x4(%a: memref<4x4xf32>, %b: memref<4x4xf32>,
                       %c: memref<4x4xf32>) attributes {vortex.entry} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<4x4xf32>)
    linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
        outs(%c : memref<4x4xf32>)
    return
  }
}

// CHECK-LABEL: llvm.func @matmul4x4(
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: !llvm.ptr
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK-NOT: linalg.matmul
// CHECK-NOT: func.func
