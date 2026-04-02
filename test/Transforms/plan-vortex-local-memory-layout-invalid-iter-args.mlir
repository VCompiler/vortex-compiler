// RUN: not vx-opt %s --vortex-plan-local-memory-layout 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %buf = vortex.local_alloc() : memref<4xf32, #vortex.address_space<local>>
    %loop = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %buf) -> (memref<4xf32, #vortex.address_space<local>>) {
      scf.yield %acc : memref<4xf32, #vortex.address_space<local>>
    }
    func.call @touch(%loop) : (memref<4xf32, #vortex.address_space<local>>) -> ()
    vortex.yield
  }
  return
}

func.func private @touch(%arg0: memref<4xf32, #vortex.address_space<local>>)

// CHECK: error:
// CHECK-SAME: cannot escape local memory planning scope through yield/branch/iter_args
