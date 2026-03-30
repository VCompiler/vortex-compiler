// RUN: vx-opt %s --vortex-map-parallel-loops-to-launch | FileCheck %s

func.func @kernel(%out: memref<8x8xi32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : i32

  scf.for %tile = %c0 to %c2 step %c1 {
    "scf.for"(%c0, %c2, %c1) ({
    ^bb0(%sg: index):
      "scf.for"(%c0, %c4, %c1) ({
      ^bb0(%th: index):
        memref.store %c7, %out[%sg, %th] : memref<8x8xi32, #vortex.address_space<global>>
        scf.yield
      }) {vortex.mapping = "thread"} : (index, index, index) -> ()
      scf.yield
    }) {vortex.mapping = "subgroup"} : (index, index, index) -> ()
  }
  return
}

// CHECK-LABEL: func.func @kernel(
// CHECK: scf.for %[[TILE:.*]] = %c0 to %c2 step %c1 {
// CHECK: vortex.launch %c1, %c2, %c4 {
// CHECK-NOT: vortex.core_id
// CHECK: %[[SG:.*]] = vortex.subgroup_id : index
// CHECK: %[[TH:.*]] = vortex.thread_id : index
// CHECK: memref.store %{{.*}}, %arg0[%[[SG]], %[[TH]]] : memref<8x8xi32, #vortex.address_space<global>>
// CHECK: }
