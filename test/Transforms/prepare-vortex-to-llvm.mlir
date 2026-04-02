// RUN: vx-opt %s --vortex-legalize-for-llvm | FileCheck %s

func.func private @touch(%arg0: memref<16xf32, #vortex.address_space<global>>) {
  return
}

func.func private @consume(%value: f32) {
  return
}

func.func @kernel(%arg0: memref<16xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 3.000000e+00 : f32

  vortex.launch %c1, %c1, %c1 {
    %tid = vortex.thread_id : index
    %idx = affine.apply affine_map<(d0) -> (d0 + 1)>(%tid)
    %private = memref.alloca() : memref<4xf32, #vortex.address_space<private>>
    memref.store %cst, %private[%c0] : memref<4xf32, #vortex.address_space<private>>
    %value = memref.load %private[%c0] : memref<4xf32, #vortex.address_space<private>>
    memref.store %cst, %arg0[%idx] : memref<16xf32, #vortex.address_space<global>>
    func.call @touch(%arg0) : (memref<16xf32, #vortex.address_space<global>>) -> ()
    vortex.barrier <core>
    func.call @consume(%value) : (f32) -> ()
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func private @touch(
// CHECK-SAME: %[[ARG0:.*]]: memref<16xf32>
// CHECK-LABEL: func.func @kernel(
// CHECK-SAME: %[[KARG0:.*]]: memref<16xf32>
// CHECK-NOT: vortex.launch
// CHECK: %[[TID:.*]] = vortex.thread_id : index
// CHECK: %[[IDX:.*]] = arith.addi %[[TID]], %{{.*}} : index
// CHECK: %[[PRIVATE:.*]] = memref.alloca() : memref<4xf32, 2>
// CHECK: memref.store %cst, %[[PRIVATE]][%c0] : memref<4xf32, 2>
// CHECK: %[[VALUE:.*]] = memref.load %[[PRIVATE]][%c0] : memref<4xf32, 2>
// CHECK: memref.store %cst, %[[KARG0]][%[[IDX]]] : memref<16xf32>
// CHECK: call @touch(%[[KARG0]]) : (memref<16xf32>) -> ()
// CHECK: vortex.barrier <core>
// CHECK: call @consume(%[[VALUE]]) : (f32) -> ()
// CHECK-NOT: affine.apply
// CHECK-NOT: #vortex.address_space<global>
// CHECK-NOT: #vortex.address_space<private>
// CHECK-NOT: #vortex.address_space<local>
