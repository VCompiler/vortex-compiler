module {
  func.func @local_memory_coop_i32(%out: memref<8xi32>) attributes {vortex.entry} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %v7 = arith.constant 7 : i32
    %v100 = arith.constant 100 : i32

    vortex.launch %c1, %c2, %c4 {
      %wid = vortex.subgroup_id : index
      %tid = vortex.thread_id : index

      %buf = vortex.local_alloc() : memref<8xi32, #vortex.address_space<local>>

      %warp_offset = arith.muli %wid, %c4 : index
      %local_slot = arith.addi %warp_offset, %tid : index

      %wid_i32 = arith.index_cast %wid : index to i32
      %tid_i32 = arith.index_cast %tid : index to i32
      %warp_term = arith.muli %wid_i32, %v100 : i32
      %partial = arith.addi %warp_term, %tid_i32 : i32
      %value = arith.addi %partial, %v7 : i32

      memref.store %value, %buf[%local_slot] : memref<8xi32, #vortex.address_space<local>>
      vortex.barrier <core>

      %peer_wid = arith.subi %c1, %wid : index
      %peer_offset = arith.muli %peer_wid, %c4 : index
      %peer_slot = arith.addi %peer_offset, %tid : index
      %readback = memref.load %buf[%peer_slot] : memref<8xi32, #vortex.address_space<local>>
      memref.store %readback, %out[%local_slot] : memref<8xi32>
      vortex.yield
    }
    return
  }
}
