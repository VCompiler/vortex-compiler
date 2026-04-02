// RUN: vx-opt %s --pass-pipeline='builtin.module(vortex-mvp-backend-pipeline)' | FileCheck %s

func.func @kernel(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 3.000000e+00 : f32

  vortex.launch %c1, %c1, %c1 {
    %tid = vortex.thread_id : index
    %idx = affine.apply affine_map<(d0) -> (d0 + 1)>(%tid)
    memref.store %cst, %arg0[%idx] : memref<16xf32, #vortex.address_space<global>>
    vortex.barrier <core>
    vortex.yield
  }
  return
}

// CHECK-DAG: llvm.func @vx_thread_id() -> i32 attributes {sym_visibility = "private"}
// CHECK-DAG: llvm.func @vx_num_warps() -> i32 attributes {sym_visibility = "private"}
// CHECK-DAG: llvm.func @vx_barrier(i32, i32) attributes {sym_visibility = "private"}
// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: attributes {vortex.kernel_entry}
// CHECK: llvm.call @vx_thread_id() : () -> i32
// CHECK: llvm.call @vx_num_warps() : () -> i32
// CHECK: llvm.call @vx_barrier
// CHECK-NOT: vortex.launch
// CHECK-NOT: vortex.thread_id
// CHECK-NOT: vortex.barrier
// CHECK-NOT: memref.store
