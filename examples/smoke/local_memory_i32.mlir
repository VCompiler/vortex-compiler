module {
  func.func @local_memory_i32(%out: memref<4xi32>) attributes {vortex.entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %v7 = arith.constant 7 : i32
    %v11 = arith.constant 11 : i32

    vortex.launch %c1, %c1, %c1 {
      %buf = vortex.local_alloc() : memref<4xi32, #vortex.address_space<local>>
      memref.store %v7, %buf[%c0] : memref<4xi32, #vortex.address_space<local>>
      memref.store %v11, %buf[%c1] : memref<4xi32, #vortex.address_space<local>>
      %load0 = memref.load %buf[%c0] : memref<4xi32, #vortex.address_space<local>>
      %load1 = memref.load %buf[%c1] : memref<4xi32, #vortex.address_space<local>>
      %sum = arith.addi %load0, %load1 : i32
      memref.store %load0, %out[%c0] : memref<4xi32>
      memref.store %load1, %out[%c1] : memref<4xi32>
      memref.store %sum, %out[%c2] : memref<4xi32>
      memref.store %v7, %out[%c3] : memref<4xi32>
      vortex.yield
    }
    return
  }
}
