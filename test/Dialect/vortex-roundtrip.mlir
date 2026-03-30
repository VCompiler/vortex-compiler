// RUN: vx-opt %s | FileCheck %s

module {
  func.func @kernel(
      %a: memref<128x128xf32, #vortex.address_space<global>>,
      %b: memref<128x128xf32, #vortex.address_space<global>>)
      attributes {vortex.kernel} {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    vortex.launch %c2, %c4, %c4 {
      %core = vortex.core_id : index
      %subgroup = vortex.subgroup_id : index
      %thread = vortex.thread_id : index
      %tile = vortex.local_alloc() : memref<16xf32, #vortex.address_space<local>>
      vortex.barrier <core>
      vortex.fence <subgroup>
      vortex.yield
    }
    return
  }
}

// CHECK: func.func @kernel
// CHECK-SAME: memref<128x128xf32, #vortex.address_space<global>>
// CHECK-SAME: attributes {vortex.kernel}
// CHECK: vortex.launch %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK: vortex.core_id : index
// CHECK: vortex.subgroup_id : index
// CHECK: vortex.thread_id : index
// CHECK: vortex.local_alloc() : memref<16xf32, #vortex.address_space<local>>
// CHECK: vortex.barrier <core>
// CHECK: vortex.fence <subgroup>
