// RUN: vx-opt %s --pass-pipeline="builtin.module(func.func(vortex-mark-kernel{remove-entry-attr=1},vortex-materialize-address-spaces,vortex-lower-linalg-inside-kernel),canonicalize,cse,vortex-legalize-for-llvm,vortex-lower-runtime-builtins,canonicalize,cse,convert-scf-to-cf,convert-math-to-llvm,convert-math-to-libm,convert-arith-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=1},convert-cf-to-llvm,reconcile-unrealized-casts)" | FileCheck %s

// Verify that math ops are lowered through the pipeline without residual math dialect ops.

module {
  func.func @math_kernel(%in: memref<8xf32>, %out: memref<8xf32>) attributes {vortex.entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %x = memref.load %in[%i] : memref<8xf32>
      %e = math.exp %x : f32
      memref.store %e, %out[%i] : memref<8xf32>
    }
    return
  }
}

// CHECK-NOT: math.exp
// CHECK-NOT: math.sqrt
// CHECK-NOT: math.erf
// CHECK: llvm.func
