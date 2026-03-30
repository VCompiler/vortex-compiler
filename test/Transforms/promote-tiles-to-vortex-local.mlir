// RUN: vx-opt %s --vortex-promote-tiles-to-local | FileCheck %s

func.func @kernel(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [8] [1] {vortex.promote_to_local} :
      memref<16xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %value = memref.load %tile[%c0] : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %sum = arith.addf %value, %value : f32
    func.call @use(%sum) : (f32) -> ()
    vortex.yield
  }
  return
}

func.func private @use(%value: f32)

// CHECK-LABEL: func.func @kernel(
// CHECK: vortex.launch %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK: %[[TILE:.*]] = memref.subview %arg0[%c0] [8] [1] : memref<16xf32, #vortex.address_space<global>> to memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
// CHECK: %[[LOCAL:.*]] = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
// CHECK-NEXT: memref.copy %[[TILE]], %[[LOCAL]] : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to memref<8xf32, #vortex.address_space<local>>
// CHECK: %[[VALUE:.*]] = memref.load %[[LOCAL]][%c0] : memref<8xf32, #vortex.address_space<local>>
// CHECK: %[[SUM:.*]] = arith.addf %[[VALUE]], %[[VALUE]] : f32
// CHECK: func.call @use(%[[SUM]]) : (f32) -> ()
// CHECK-NOT: vortex.promote_to_local
