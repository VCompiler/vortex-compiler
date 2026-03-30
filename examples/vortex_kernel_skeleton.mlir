module {
  func.func @tiled_matmul_kernel(%arg0: memref<128x128xf32, #vortex.address_space<global>>, %arg1: memref<128x128xf32, #vortex.address_space<global>>, %arg2: memref<128x128xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    vortex.launch %c1, %c4, %c4 {
      %0 = vortex.core_id : index
      %1 = vortex.subgroup_id : index
      %2 = vortex.thread_id : index
      %3 = vortex.local_alloc() : memref<8x8xf32, #vortex.address_space<local>>
      vortex.barrier <core>
      vortex.fence <subgroup>
    }
    return
  }
}

