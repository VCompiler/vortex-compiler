// RUN: vx-opt %s --allow-unregistered-dialect --vortex-tile-matmul-for-pre-vortex='tile-size=8 block-m=8 block-n=4 block-k=4 num-subgroups=2 num-threads=4' | FileCheck %s

func.func @main_graph(%arg0: memref<16x16xf32>,
                      %arg1: memref<16x16xf32>,
                      %arg2: memref<16x16xf32>) attributes {vortex.entry} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
  linalg.matmul ins(%arg0, %arg1 : memref<16x16xf32>, memref<16x16xf32>)
      outs(%alloc : memref<16x16xf32>)
  memref.copy %alloc, %arg2 : memref<16x16xf32> to memref<16x16xf32>
  return
}

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[C8:.+]] = arith.constant 8 : index
// CHECK: %[[C4:.+]] = arith.constant 4 : index
// CHECK: scf.for {{.*}} step %[[C8]] {
// CHECK: scf.for {{.*}} step %{{.*}} {
// CHECK: %[[CTILE:.+]] = memref.subview %arg2{{.*}} [8, 4] [1, 1]
// CHECK: memref.store %cst, %[[CTILE]]
// CHECK: } {vortex.mapping = "thread"}
// CHECK: } {vortex.mapping = "subgroup"}
// CHECK: scf.for {{.*}} step %{{.*}} {
// CHECK: %[[ATILE:.+]] = memref.subview %arg0{{.*}} [8, 4] [1, 1] {vortex.promote_to_local}
// CHECK: %[[BTILE:.+]] = memref.subview %arg1{{.*}} [4, 4] [1, 1] {vortex.promote_to_local}
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store {{.*}}, %[[CTILE]]
// CHECK: } {vortex.mapping = "thread"}
// CHECK: } {vortex.mapping = "subgroup"}
// CHECK: } {vortex.matmul_schedule =
// CHECK-SAME: block_k = 4 : i64
// CHECK-SAME: block_m = 8 : i64
// CHECK-SAME: block_n = 4 : i64
// CHECK-SAME: compute_policy = "linear_tid_2d"
// CHECK-SAME: copy_policy = "linear_stride"
// CHECK-SAME: num_subgroups = 2 : i64
// CHECK-SAME: num_threads = 4 : i64
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.matmul
