// RUN: not vx-opt %s --vortex-lower-local-memory 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel, vortex.local_frame_bytes = 64 : i64} {
  %c0 = arith.constant 0 : index
  %buf = vortex.local_alloc()
      {vortex.local.byte_offset = 0 : i64,
       vortex.local.byte_size = 64 : i64,
       vortex.local.alignment = 4 : i64}
      : memref<4x4xf32, #vortex.address_space<local>>
  %t = memref.transpose %buf (i, j) -> (j, i) :
    memref<4x4xf32, #vortex.address_space<local>> to
    memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>,
           #vortex.address_space<local>>
  %value = memref.load %t[%c0, %c0] :
    memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>,
           #vortex.address_space<local>>
  func.call @use(%value) : (f32) -> ()
  return
}

func.func private @use(%arg0: f32)

// CHECK: error:
// CHECK-SAME: lowering local memref through 'memref.transpose' is not implemented yet in the current MVP
