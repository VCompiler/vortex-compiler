// RUN: vx-opt %s --vortex-insert-barriers --vortex-insert-barriers | FileCheck %s

func.func @read_only_tile(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [8] [1] :
      memref<16xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %local = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
    memref.copy %tile, %local :
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to
      memref<8xf32, #vortex.address_space<local>>
    %value = memref.load %local[%c0] : memref<8xf32, #vortex.address_space<local>>
    func.call @use(%value) : (f32) -> ()
    vortex.yield
  }
  return
}

func.func @write_back_tile(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.000000e+00 : f32

  vortex.launch %c1, %c1, %c1 {
    %tile = memref.subview %arg0[%c0] [8] [1] :
      memref<16xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %local = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
    memref.copy %tile, %local :
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to
      memref<8xf32, #vortex.address_space<local>>
    memref.store %cst, %local[%c0] : memref<8xf32, #vortex.address_space<local>>
    memref.copy %local, %tile :
      memref<8xf32, #vortex.address_space<local>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

func.func @contiguous_write_backs(
    %arg0: memref<32xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 3.000000e+00 : f32

  vortex.launch %c1, %c1, %c1 {
    %tile0 = memref.subview %arg0[%c0] [8] [1] :
      memref<32xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %local0 = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
    memref.copy %tile0, %local0 :
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to
      memref<8xf32, #vortex.address_space<local>>

    %tile1 = memref.subview %arg0[%c8] [8] [1] :
      memref<32xf32, #vortex.address_space<global>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    %local1 = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
    memref.copy %tile1, %local1 :
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to
      memref<8xf32, #vortex.address_space<local>>

    memref.store %cst, %local0[%c0] : memref<8xf32, #vortex.address_space<local>>
    memref.store %cst, %local1[%c0] : memref<8xf32, #vortex.address_space<local>>
    memref.copy %local0, %tile0 :
      memref<8xf32, #vortex.address_space<local>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    memref.copy %local1, %tile1 :
      memref<8xf32, #vortex.address_space<local>> to
      memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

func.func private @use(%value: f32)

// CHECK-LABEL: func.func @read_only_tile(
// CHECK: %[[LOCAL:.*]] = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
// CHECK-NEXT: memref.copy %{{.*}}, %[[LOCAL]] : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to memref<8xf32, #vortex.address_space<local>>
// CHECK-NEXT: vortex.barrier <core>
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[LOCAL]][%c0] : memref<8xf32, #vortex.address_space<local>>
// CHECK: func.call @use(%[[VALUE]]) : (f32) -> ()

// CHECK-LABEL: func.func @write_back_tile(
// CHECK: %[[LOCAL_WB:.*]] = vortex.local_alloc() : memref<8xf32, #vortex.address_space<local>>
// CHECK-NEXT: memref.copy %{{.*}}, %[[LOCAL_WB]] : memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>> to memref<8xf32, #vortex.address_space<local>>
// CHECK-NEXT: vortex.barrier <core>
// CHECK-NEXT: memref.store %cst, %[[LOCAL_WB]][%c0] : memref<8xf32, #vortex.address_space<local>>
// CHECK-NEXT: vortex.barrier <core>
// CHECK-NEXT: memref.copy %[[LOCAL_WB]], %{{.*}} : memref<8xf32, #vortex.address_space<local>> to memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>

// CHECK-LABEL: func.func @contiguous_write_backs(
// CHECK: memref.store %cst, %[[LOCAL0:.*]][%c0] : memref<8xf32, #vortex.address_space<local>>
// CHECK: memref.store %cst, %[[LOCAL1:.*]][%c0] : memref<8xf32, #vortex.address_space<local>>
// CHECK: vortex.barrier <core>
// CHECK-NEXT: memref.copy %[[LOCAL0]], %{{.*}} : memref<8xf32, #vortex.address_space<local>> to memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
// CHECK-NEXT: memref.copy %[[LOCAL1]], %{{.*}} : memref<8xf32, #vortex.address_space<local>> to memref<8xf32, strided<[1], offset: ?>, #vortex.address_space<global>>
