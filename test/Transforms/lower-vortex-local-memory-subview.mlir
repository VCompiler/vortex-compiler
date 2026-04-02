// RUN: vx-opt %s --vortex-lower-local-memory | FileCheck %s

func.func @kernel_subview_copy(
    %src: memref<4xf32, #vortex.address_space<global>>,
    %dst: memref<4xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel, vortex.local_frame_bytes = 32 : i64} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %buf = vortex.local_alloc()
        {vortex.local.byte_offset = 0 : i64,
         vortex.local.byte_size = 32 : i64,
         vortex.local.alignment = 4 : i64}
        : memref<8xf32, #vortex.address_space<local>>
    %sub = memref.subview %buf[%c1] [4] [2] :
      memref<8xf32, #vortex.address_space<local>> to
      memref<4xf32, strided<[2], offset: ?>, #vortex.address_space<local>>
    memref.copy %src, %sub :
      memref<4xf32, #vortex.address_space<global>> to
      memref<4xf32, strided<[2], offset: ?>, #vortex.address_space<local>>
    memref.copy %sub, %dst :
      memref<4xf32, strided<[2], offset: ?>, #vortex.address_space<local>> to
      memref<4xf32, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func private @vx_local_mem_base() -> i64
// CHECK-LABEL: func.func @kernel_subview_copy(
// CHECK: %[[BASE:.*]] = call @vx_local_mem_base() : () -> i64
// CHECK: vortex.launch
// CHECK: scf.for
// CHECK: llvm.inttoptr
// CHECK: llvm.store
// CHECK: scf.for
// CHECK: llvm.load
// CHECK: memref.store

func.func @kernel_nested_rank_reduced_subview(
    %src: memref<4x4xf32, #vortex.address_space<global>>,
    %dst: memref<2xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel, vortex.local_frame_bytes = 64 : i64} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %buf = vortex.local_alloc()
        {vortex.local.byte_offset = 0 : i64,
         vortex.local.byte_size = 64 : i64,
         vortex.local.alignment = 4 : i64}
        : memref<4x4xf32, #vortex.address_space<local>>
    memref.copy %src, %buf :
      memref<4x4xf32, #vortex.address_space<global>> to
      memref<4x4xf32, #vortex.address_space<local>>
    %row = memref.subview %buf[%c1, %c0] [1, 4] [1, 1] :
      memref<4x4xf32, #vortex.address_space<local>> to
      memref<1x4xf32, strided<[4, 1], offset: ?>, #vortex.address_space<local>>
    %slice = memref.subview %row[%c0, %c1] [1, 2] [1, 2] :
      memref<1x4xf32, strided<[4, 1], offset: ?>, #vortex.address_space<local>> to
      memref<2xf32, strided<[2], offset: ?>, #vortex.address_space<local>>
    %v0 = memref.load %slice[%c0] :
      memref<2xf32, strided<[2], offset: ?>, #vortex.address_space<local>>
    %v1 = memref.load %slice[%c1] :
      memref<2xf32, strided<[2], offset: ?>, #vortex.address_space<local>>
    memref.store %v0, %dst[%c0] :
      memref<2xf32, #vortex.address_space<global>>
    memref.store %v1, %dst[%c1] :
      memref<2xf32, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @kernel_nested_rank_reduced_subview(
// CHECK: %[[BASE2:.*]] = call @vx_local_mem_base() : () -> i64
// CHECK: vortex.launch
// CHECK: scf.for
// CHECK: llvm.store
// CHECK: llvm.load
// CHECK: llvm.load
// CHECK: memref.store
// CHECK: memref.store
// CHECK-NOT: memref.subview
// CHECK-NOT: vortex.local_alloc
// CHECK-NOT: memref.copy
// CHECK-NOT: #vortex.address_space<local>
