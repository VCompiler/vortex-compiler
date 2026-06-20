// RUN: vx-opt %s --vortex-fuse-linear-with-bias | FileCheck %s

func.func @linear_with_bias(
    %lhs: memref<2x3xf32, #vortex.address_space<global>>,
    %rhs: memref<4x3xf32, #vortex.address_space<global>>,
    %bias: memref<4xf32, #vortex.address_space<global>>,
    %out: memref<2x4xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %zero = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%zero : f32) outs(%out : memref<2x4xf32, #vortex.address_space<global>>)
  linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (j, k)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : memref<2x3xf32, #vortex.address_space<global>>, memref<4x3xf32, #vortex.address_space<global>>)
    outs(%out : memref<2x4xf32, #vortex.address_space<global>>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %acc: f32):
    %prod = arith.mulf %lhs_value, %rhs_value : f32
    %sum = arith.addf %prod, %acc : f32
    linalg.yield %sum : f32
  }
  linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%out, %bias : memref<2x4xf32, #vortex.address_space<global>>, memref<4xf32, #vortex.address_space<global>>)
    outs(%out : memref<2x4xf32, #vortex.address_space<global>>) {
  ^bb0(%value: f32, %bias_value: f32, %dummy: f32):
    %result = arith.addf %value, %bias_value : f32
    linalg.yield %result : f32
  }
  return
}

func.func @linear_without_bias(
    %lhs: memref<2x3xf32, #vortex.address_space<global>>,
    %rhs: memref<4x3xf32, #vortex.address_space<global>>,
    %out: memref<2x4xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %zero = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%zero : f32) outs(%out : memref<2x4xf32, #vortex.address_space<global>>)
  linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (j, k)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : memref<2x3xf32, #vortex.address_space<global>>, memref<4x3xf32, #vortex.address_space<global>>)
    outs(%out : memref<2x4xf32, #vortex.address_space<global>>) {
  ^bb0(%lhs_value: f32, %rhs_value: f32, %acc: f32):
    %prod = arith.mulf %lhs_value, %rhs_value : f32
    %sum = arith.addf %prod, %acc : f32
    linalg.yield %sum : f32
  }
  return
}

// CHECK-LABEL: func.func @linear_with_bias(
// CHECK-SAME: %[[LHS:.*]]: memref<2x3xf32, #vortex.address_space<global>>
// CHECK-SAME: %[[RHS:.*]]: memref<4x3xf32, #vortex.address_space<global>>
// CHECK-SAME: %[[BIAS:.*]]: memref<4xf32, #vortex.address_space<global>>
// CHECK-SAME: %[[OUT:.*]]: memref<2x4xf32, #vortex.address_space<global>>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.for %[[I:.*]] =
// CHECK: scf.for %[[J:.*]] =
// CHECK: %[[SUM:.*]] = scf.for %[[K:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (f32) {
// CHECK: %[[A:.*]] = memref.load %[[LHS]][%[[I]], %[[K]]] : memref<2x3xf32, #vortex.address_space<global>>
// CHECK: %[[B:.*]] = memref.load %[[RHS]][%[[J]], %[[K]]] : memref<4x3xf32, #vortex.address_space<global>>
// CHECK: %[[PROD:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK: %[[NEXT:.*]] = arith.addf %[[PROD]], %[[ACC]] : f32
// CHECK: scf.yield %[[NEXT]] : f32
// CHECK: %[[BIAS_VALUE:.*]] = memref.load %[[BIAS]][%[[J]]] : memref<4xf32, #vortex.address_space<global>>
// CHECK: %[[RESULT:.*]] = arith.addf %[[SUM]], %[[BIAS_VALUE]] : f32
// CHECK: memref.store %[[RESULT]], %[[OUT]][%[[I]], %[[J]]] : memref<2x4xf32, #vortex.address_space<global>>
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.generic

// CHECK-LABEL: func.func @linear_without_bias(
// CHECK-SAME: %[[LHS2:.*]]: memref<2x3xf32, #vortex.address_space<global>>
// CHECK-SAME: %[[RHS2:.*]]: memref<4x3xf32, #vortex.address_space<global>>
// CHECK-SAME: %[[OUT2:.*]]: memref<2x4xf32, #vortex.address_space<global>>
// CHECK: %[[ZERO2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.for %[[I2:.*]] =
// CHECK: scf.for %[[J2:.*]] =
// CHECK: %[[SUM2:.*]] = scf.for %[[K2:.*]] = {{.*}} iter_args(%[[ACC2:.*]] = %[[ZERO2]]) -> (f32) {
// CHECK: %[[A2:.*]] = memref.load %[[LHS2]][%[[I2]], %[[K2]]] : memref<2x3xf32, #vortex.address_space<global>>
// CHECK: %[[B2:.*]] = memref.load %[[RHS2]][%[[J2]], %[[K2]]] : memref<4x3xf32, #vortex.address_space<global>>
// CHECK: %[[PROD2:.*]] = arith.mulf %[[A2]], %[[B2]] : f32
// CHECK: %[[NEXT2:.*]] = arith.addf %[[PROD2]], %[[ACC2]] : f32
// CHECK: scf.yield %[[NEXT2]] : f32
// CHECK: memref.store %[[SUM2]], %[[OUT2]][%[[I2]], %[[J2]]] : memref<2x4xf32, #vortex.address_space<global>>
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.generic
