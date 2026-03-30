// RUN: not vx-opt %s --vortex-promote-tiles-to-local 2>&1 | FileCheck %s

func.func @kernel(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [8] [1] {vortex.promote_to_local} :
      memref<16xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %loop = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %tile) -> (memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>) {
      scf.yield %acc : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    }
    func.call @use_memref(%loop) : (memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>) -> ()
    vortex.yield
  }
  return
}

func.func private @use_memref(%value: memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>)

// CHECK: error:
// CHECK-SAME: cannot escape promotion scope through yield/branch/iter_args
