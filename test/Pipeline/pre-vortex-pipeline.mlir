// RUN: vx-opt %s --pass-pipeline='builtin.module(vortex-pre-vortex-pipeline)' | FileCheck %s

func.func @vector_copy(%src: memref<4xf32>, %dst: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %vec = vector.transfer_read %src[%c0], %pad
      : memref<4xf32>, vector<4xf32>
  vector.transfer_write %vec, %dst[%c0]
      : vector<4xf32>, memref<4xf32>
  return
}

// CHECK: func.func @vector_copy
// CHECK: vortex.pre_vortex_dialects = ["vector"]
// CHECK: vortex.pre_vortex_memory_spaces = ["<default>"]
// CHECK: vortex.pre_vortex_ops = ["vector.transfer_read", "vector.transfer_write"]
