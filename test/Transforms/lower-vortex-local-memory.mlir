// RUN: vx-opt %s --vortex-lower-local-memory | FileCheck %s

func.func @kernel(
    %src: memref<4xf32, #vortex.address_space<global>>,
    %dst: memref<4xf32, #vortex.address_space<global>>,
    %dst2: memref<4xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel, vortex.local_frame_bytes = 16 : i64} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %buf = vortex.local_alloc()
        {vortex.local.byte_offset = 0 : i64,
         vortex.local.byte_size = 16 : i64,
         vortex.local.alignment = 4 : i64}
        : memref<4xf32, #vortex.address_space<local>>
    memref.copy %src, %buf : memref<4xf32, #vortex.address_space<global>> to memref<4xf32, #vortex.address_space<local>>
    %cast = memref.cast %buf : memref<4xf32, #vortex.address_space<local>> to memref<?xf32, #vortex.address_space<local>>
    %value = memref.load %cast[%c0] : memref<?xf32, #vortex.address_space<local>>
    memref.store %value, %dst[%c0] : memref<4xf32, #vortex.address_space<global>>
    memref.copy %buf, %dst2 : memref<4xf32, #vortex.address_space<local>> to memref<4xf32, #vortex.address_space<global>>
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func private @vx_local_mem_base() -> i64
// CHECK-LABEL: func.func @kernel(
// CHECK: %[[BASE:.*]] = call @vx_local_mem_base() : () -> i64
// CHECK: vortex.launch
// CHECK: scf.for
// CHECK: memref.load %arg0
// CHECK: %[[PTR0:.*]] = llvm.inttoptr
// CHECK: llvm.store %{{.*}}, %[[PTR0]] {alignment = 4 : i64} : f32, !llvm.ptr
// CHECK: %[[PTR1:.*]] = llvm.inttoptr
// CHECK: %[[VAL:.*]] = llvm.load %[[PTR1]] {alignment = 4 : i64} : !llvm.ptr -> f32
// CHECK: memref.store %[[VAL]], %arg1[%c0] : memref<4xf32, #vortex.address_space<global>>
// CHECK: scf.for
// CHECK: llvm.load
// CHECK: memref.store
// CHECK-NOT: vortex.local_alloc
// CHECK-NOT: memref.cast
// CHECK-NOT: memref.copy
// CHECK-NOT: #vortex.address_space<local>
