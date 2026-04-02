// RUN: not vx-opt %s --vortex-lower-runtime-builtins 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  %buf = vortex.local_alloc() : memref<4xf32, #vortex.address_space<local>>
  return
}

// CHECK: error: 'vortex.local_alloc' op vortex.local_alloc lowering is not implemented yet in vortex-lower-runtime-builtins
