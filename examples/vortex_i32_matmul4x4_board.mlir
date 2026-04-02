module {
  func.func @main() -> i32 attributes {vortex.kernel} {
    %cid = vortex.core_id : index
    %wid = vortex.subgroup_id : index
    %tid = vortex.thread_id : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %ret0 = arith.constant 0 : i32

    %addrA = llvm.mlir.constant(2164326400 : i64) : i64
    %addrB = llvm.mlir.constant(2164330496 : i64) : i64
    %addrC = llvm.mlir.constant(2164334592 : i64) : i64
    %baseA = llvm.inttoptr %addrA : i64 to !llvm.ptr
    %baseB = llvm.inttoptr %addrB : i64 to !llvm.ptr
    %baseC = llvm.inttoptr %addrC : i64 to !llvm.ptr

    %isCore0 = arith.cmpi eq, %cid, %c0 : index
    %isWarp0 = arith.cmpi eq, %wid, %c0 : index
    %isThread0 = arith.cmpi eq, %tid, %c0 : index
    %isCoreWarp0 = arith.andi %isCore0, %isWarp0 : i1
    %shouldRun = arith.andi %isCoreWarp0, %isThread0 : i1

    scf.if %shouldRun {
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (i32) {
            %aRowBase = arith.muli %i, %c4 : index
            %aIndex = arith.addi %aRowBase, %k : index
            %aIndexI32 = arith.index_cast %aIndex : index to i32
            %aPtr = llvm.getelementptr %baseA[%aIndexI32] : (!llvm.ptr, i32) -> !llvm.ptr, i32
            %aVal = llvm.load %aPtr : !llvm.ptr -> i32

            %bRowBase = arith.muli %k, %c4 : index
            %bIndex = arith.addi %bRowBase, %j : index
            %bIndexI32 = arith.index_cast %bIndex : index to i32
            %bPtr = llvm.getelementptr %baseB[%bIndexI32] : (!llvm.ptr, i32) -> !llvm.ptr, i32
            %bVal = llvm.load %bPtr : !llvm.ptr -> i32

            %prod = arith.muli %aVal, %bVal : i32
            %next = arith.addi %acc, %prod : i32
            scf.yield %next : i32
          }

          %cRowBase = arith.muli %i, %c4 : index
          %cIndex = arith.addi %cRowBase, %j : index
          %cIndexI32 = arith.index_cast %cIndex : index to i32
          %cPtr = llvm.getelementptr %baseC[%cIndexI32] : (!llvm.ptr, i32) -> !llvm.ptr, i32
          llvm.store %sum, %cPtr : i32, !llvm.ptr
        }
      }
    }

    return %ret0 : i32
  }
}
