module {
  func.func @local_memory_multi_i32(%out: memref<16xi32>) attributes {vortex.entry} {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %v7 = arith.constant 7 : i32
    %v100 = arith.constant 100 : i32
    %v1000 = arith.constant 1000 : i32

    vortex.launch %c2, %c2, %c4 {
      %cid = vortex.core_id : index
      %wid = vortex.subgroup_id : index
      %tid = vortex.thread_id : index

      %buf = vortex.local_alloc() : memref<8xi32, #vortex.address_space<local>>

      %warp_offset = arith.muli %wid, %c4 : index
      %local_slot = arith.addi %warp_offset, %tid : index
      %core_offset = arith.muli %cid, %c8 : index
      %global_slot = arith.addi %core_offset, %local_slot : index

      %cid_i32 = arith.index_cast %cid : index to i32
      %wid_i32 = arith.index_cast %wid : index to i32
      %tid_i32 = arith.index_cast %tid : index to i32
      %core_term = arith.muli %cid_i32, %v1000 : i32
      %warp_term = arith.muli %wid_i32, %v100 : i32
      %partial = arith.addi %core_term, %warp_term : i32
      %partial2 = arith.addi %partial, %tid_i32 : i32
      %value = arith.addi %partial2, %v7 : i32

      memref.store %value, %buf[%local_slot] : memref<8xi32, #vortex.address_space<local>>
      %readback = memref.load %buf[%local_slot] : memref<8xi32, #vortex.address_space<local>>
      memref.store %readback, %out[%global_slot] : memref<16xi32>
      vortex.yield
    }
    return
  }
}
