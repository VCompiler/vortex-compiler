// RUN: not vx-opt %s --vortex-lower-runtime-builtins 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  vortex.launch %c1, %c1, %c1 {
    vortex.yield
  }
  return
}

// CHECK: requires running vortex-legalize-for-llvm before vortex-lower-runtime-builtins
