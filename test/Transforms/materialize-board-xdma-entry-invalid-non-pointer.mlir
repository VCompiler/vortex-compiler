// RUN: not vx-opt %s --vortex-materialize-board-xdma-entry 2>&1 | FileCheck %s

module {
  llvm.func @kernel(%arg0: i32) attributes {vortex.kernel_entry} {
    llvm.return
  }
}

// CHECK: board/XDMA entry wrapper currently supports only bare-pointer lowered kernel arguments
