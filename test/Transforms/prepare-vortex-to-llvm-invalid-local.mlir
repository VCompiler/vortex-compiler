// RUN: not vx-opt %s --vortex-legalize-for-llvm 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<4xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  vortex.launch %c1, %c1, %c1 {
    %buf = vortex.local_alloc() : memref<4xf32, #vortex.address_space<local>>
    %v = memref.load %arg0[%c0] : memref<4xf32, #vortex.address_space<global>>
    memref.store %v, %buf[%c0] : memref<4xf32, #vortex.address_space<local>>
    vortex.yield
  }
  return
}

// CHECK: error:
// CHECK-SAME: requires running vortex-lower-local-memory before vortex-legalize-for-llvm
