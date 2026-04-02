// RUN: vx-opt %s --vortex-materialize-address-spaces | FileCheck %s

func.func @kernel(%arg0: memref<16x16xf32>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %tile = memref.subview %arg0[%c0, %c0] [8, 8] [1, 1] :
    memref<16x16xf32> to memref<8x8xf32, strided<[16, 1], offset: ?>>
  %cast = memref.cast %tile :
    memref<8x8xf32, strided<[16, 1], offset: ?>> to
    memref<?x?xf32, strided<[?, ?], offset: ?>>
  %value = memref.load %cast[%c0, %c0] :
    memref<?x?xf32, strided<[?, ?], offset: ?>>
  %sum = arith.addf %value, %value : f32
  memref.store %sum, %arg0[%c8, %c8] :
    memref<16x16xf32>
  return
}

// CHECK-LABEL: func.func @kernel(
// CHECK-SAME: %[[ARG0:.*]]: memref<16x16xf32, #vortex.address_space<global>>
// CHECK: %[[TILE:.*]] = memref.subview %[[ARG0]][%c0, %c0] [8, 8] [1, 1] :
// CHECK-SAME: memref<16x16xf32, #vortex.address_space<global>>
// CHECK-SAME: to memref<8x8xf32, strided<[16, 1], offset: ?>, #vortex.address_space<global>>
// CHECK: %[[CAST:.*]] = memref.cast %[[TILE]] :
// CHECK-SAME: memref<8x8xf32, strided<[16, 1], offset: ?>, #vortex.address_space<global>>
// CHECK-SAME: to memref<?x?xf32, strided<[?, ?], offset: ?>, #vortex.address_space<global>>
