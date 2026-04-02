// RUN: vx-opt %s --vortex-plan-local-memory-layout | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index

  vortex.launch %c1, %c1, %c1 {
    %buf0 = vortex.local_alloc() : memref<3xf32, #vortex.address_space<local>>
    %buf1 = vortex.local_alloc() : memref<2xf64, #vortex.address_space<local>>
    %buf2 = vortex.local_alloc() : memref<5xi8, #vortex.address_space<local>>
    memref.copy %buf2, %buf2 : memref<5xi8, #vortex.address_space<local>> to memref<5xi8, #vortex.address_space<local>>
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @kernel() attributes {
// CHECK-SAME: vortex.kernel
// CHECK-SAME: vortex.local_frame_bytes = 37 : i64
// CHECK: %[[BUF0:.*]] = vortex.local_alloc() {vortex.local.alignment = 4 : i64, vortex.local.byte_offset = 0 : i64, vortex.local.byte_size = 12 : i64} : memref<3xf32, #vortex.address_space<local>>
// CHECK: %[[BUF1:.*]] = vortex.local_alloc() {vortex.local.alignment = 8 : i64, vortex.local.byte_offset = 16 : i64, vortex.local.byte_size = 16 : i64} : memref<2xf64, #vortex.address_space<local>>
// CHECK: %[[BUF2:.*]] = vortex.local_alloc() {vortex.local.alignment = 1 : i64, vortex.local.byte_offset = 32 : i64, vortex.local.byte_size = 5 : i64} : memref<5xi8, #vortex.address_space<local>>
