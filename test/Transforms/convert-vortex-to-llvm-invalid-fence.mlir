// RUN: not vx-opt %s --vortex-lower-runtime-builtins 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  vortex.fence <core>
  return
}

// CHECK: error: 'vortex.fence' op vortex.fence lowering is not implemented yet in vortex-lower-runtime-builtins
