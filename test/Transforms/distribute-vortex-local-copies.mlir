// RUN: vx-opt %s --vortex-distribute-local-copies | FileCheck %s

func.func @kernel_1d(
    %src: memref<8xf32, #vortex.address_space<global>>,
    %dst: memref<8xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  vortex.launch %c1, %c2, %c4 {
    %src_tile = memref.subview %src[%c0] [8] [1] :
      memref<8xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %dst_tile = memref.subview %dst[%c0] [8] [1] :
      memref<8xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %local = vortex.local_alloc() :
      memref<8xf32, #vortex.address_space<local>>

    memref.copy %src_tile, %local :
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to
      memref<8xf32, #vortex.address_space<local>>
    vortex.barrier <core>

    %value = memref.load %local[%c0] :
      memref<8xf32, #vortex.address_space<local>>
    memref.store %value, %local[%c0] :
      memref<8xf32, #vortex.address_space<local>>

    vortex.barrier <core>
    memref.copy %local, %dst_tile :
      memref<8xf32, #vortex.address_space<local>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @kernel_1d
// CHECK: %[[LOCAL:.*]] = vortex.local_alloc
// CHECK-NOT: memref.copy
// CHECK: %[[WID:.*]] = vortex.subgroup_id : index
// CHECK: %[[TID:.*]] = vortex.thread_id : index
// CHECK: %[[WARP_BASE:.*]] = arith.muli %[[WID]], %c4 : index
// CHECK: %[[LINEAR:.*]] = arith.addi %[[WARP_BASE]], %[[TID]] : index
// CHECK: %[[LANES:.*]] = arith.muli %c2, %c4 : index
// CHECK: scf.for %[[I:.*]] = %[[LINEAR]] to %c8 step %[[LANES]] {
// CHECK:   %[[X:.*]] = memref.load %{{.*}}[%[[I]]]
// CHECK:   memref.store %[[X]], %[[LOCAL]][%[[I]]]
// CHECK: vortex.barrier <core>
// CHECK: memref.load %[[LOCAL]][%c0]
// CHECK: memref.store %{{.*}}, %[[LOCAL]][%c0]
// CHECK: vortex.barrier <core>
// CHECK-NOT: memref.copy
// CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[Y:.*]] = memref.load %[[LOCAL]][%[[J]]]
// CHECK:   memref.store %[[Y]], %{{.*}}[%[[J]]]
// CHECK-NOT: memref.copy
// CHECK: return

func.func @kernel_2d(
    %src: memref<4x8xf32, #vortex.address_space<global>>,
    %dst: memref<4x8xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  vortex.launch %c1, %c2, %c4 {
    %src_tile = memref.subview %src[%c0, %c0] [2, 4] [1, 1] :
      memref<4x8xf32, #vortex.address_space<global>> to
      memref<2x4xf32, strided<[8, 1], offset: ?>, #vortex.address_space<global>>
    %dst_tile = memref.subview %dst[%c0, %c0] [2, 4] [1, 1] :
      memref<4x8xf32, #vortex.address_space<global>> to
      memref<2x4xf32, strided<[8, 1], offset: ?>, #vortex.address_space<global>>
    %local = vortex.local_alloc() :
      memref<2x4xf32, #vortex.address_space<local>>

    memref.copy %src_tile, %local :
      memref<2x4xf32, strided<[8, 1], offset: ?>, #vortex.address_space<global>> to
      memref<2x4xf32, #vortex.address_space<local>>
    memref.copy %local, %dst_tile :
      memref<2x4xf32, #vortex.address_space<local>> to
      memref<2x4xf32, strided<[8, 1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @kernel_2d
// CHECK-NOT: memref.copy
// CHECK: scf.for %[[I2:.*]] = %{{.*}} to %c8 step %{{.*}} {
// CHECK:   %[[ROW_BASE:.*]] = arith.divui %[[I2]], %{{.*}} : index
// CHECK:   %[[ROW:.*]] = arith.remui %[[ROW_BASE]], %{{.*}} : index
// CHECK:   %[[COL:.*]] = arith.remui %[[I2]], %{{.*}} : index
// CHECK:   %[[X2:.*]] = memref.load %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK:   memref.store %[[X2]], %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NOT: memref.copy
// CHECK: scf.for %[[J2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %{{.*}} = arith.divui %[[J2]], %{{.*}} : index
// CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK-NOT: memref.copy
// CHECK: return
