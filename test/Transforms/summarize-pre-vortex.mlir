// RUN: vx-opt %s --vortex-validate-pre-vortex --vortex-summarize-pre-vortex | FileCheck %s

func.func @stream(%src: memref<16xf32>, %dst: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %value = memref.load %src[%i] : memref<16xf32>
    %twice = arith.addf %value, %value : f32
    memref.store %twice, %dst[%i] : memref<16xf32>
  }
  return
}

// CHECK: func.func @stream
// CHECK: vortex.pre_vortex_dialects = ["arith", "memref", "scf"]
// CHECK: vortex.pre_vortex_memory_spaces = ["<default>"]
// CHECK: vortex.pre_vortex_ops = ["arith.addf", "memref.load", "memref.store", "scf.for"]
