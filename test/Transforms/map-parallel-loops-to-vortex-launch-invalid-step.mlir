// RUN: not vx-opt %s --vortex-map-parallel-loops-to-launch 2>&1 | FileCheck %s

func.func @kernel(%out: memref<8x8xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : i32

  "scf.for"(%c0, %c4, %c2) ({
  ^bb0(%th: index):
    memref.store %c7, %out[%th, %th] : memref<8x8xi32, #vortex.address_space<global>>
    scf.yield
  }) {vortex.mapping = "thread"} : (index, index, index) -> ()
  return
}

// CHECK: error:
// CHECK-SAME: mapped loops must have step 1
