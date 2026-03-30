// RUN: vx-opt %s --vortex-mark-kernel='kernel-name=tiled_matmul' --vortex-materialize-address-spaces | FileCheck %s

func.func @helper(%arg0: memref<4xf32>, %arg1: index) {
  return
}

func.func @tiled_matmul(%arg0: memref<4xf32>, %arg1: index,
                        %arg2: memref<8x?xf32, #vortex.address_space<global>>)
    attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %0 = memref.load %arg0[%c0] : memref<4xf32>
  return
}

// CHECK-LABEL: func.func @helper(
// CHECK-SAME: %[[ARG0:.*]]: memref<4xf32>, %[[ARG1:.*]]: index

// CHECK-LABEL: func.func @tiled_matmul(
// CHECK-SAME: %[[KARG0:.*]]: memref<4xf32, #vortex.address_space<global>>
// CHECK-SAME: %[[KARG1:.*]]: index
// CHECK-SAME: %[[KARG2:.*]]: memref<8x?xf32, #vortex.address_space<global>>
// CHECK-SAME: attributes {vortex.kernel}
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL:.*]] = memref.load %[[KARG0]][%[[C0]]] : memref<4xf32, #vortex.address_space<global>>
