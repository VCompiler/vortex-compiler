// RUN: vx-opt %s --vortex-promote-tiles-to-local | FileCheck %s

func.func @kernel(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.000000e+00 : f32

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [8] [1]
      {vortex.promote_to_local, vortex.write_back} :
      memref<16xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    memref.store %cst, %tile[%c0] : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @kernel(
// CHECK: vortex.launch %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK: %[[TILE:.*]] = memref.subview %arg0[%c0] [8] [1] : memref<16xf32, #vortex.address_space<global>> to memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
// CHECK: %[[LOCAL:.*]] = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
// CHECK-NEXT: memref.copy %[[TILE]], %[[LOCAL]] : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to memref<8xf32, #vortex.address_space<local>>
// CHECK: memref.store %cst, %[[LOCAL]][%c0] : memref<8xf32, #vortex.address_space<local>>
// CHECK: memref.copy %[[LOCAL]], %[[TILE]] : memref<8xf32, #vortex.address_space<local>> to memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
// CHECK-NOT: vortex.write_back
