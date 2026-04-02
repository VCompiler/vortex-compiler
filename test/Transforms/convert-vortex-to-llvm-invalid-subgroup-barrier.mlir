// RUN: not vx-opt %s --vortex-lower-runtime-builtins 2>&1 | FileCheck %s

func.func @kernel() attributes {vortex.kernel} {
  vortex.barrier <subgroup>
  return
}

// CHECK: error: 'vortex.barrier' op only vortex.barrier <core> is supported in the current MVP
