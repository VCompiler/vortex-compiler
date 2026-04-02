// RUN: not vx-opt %s --vortex-legalize-for-llvm 2>&1 | FileCheck %s

func.func @host_only() {
  %c1 = arith.constant 1 : index
  vortex.launch %c1, %c1, %c1 {
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: requires enclosing func.func marked with vortex.kernel
