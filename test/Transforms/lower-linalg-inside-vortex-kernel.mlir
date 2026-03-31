// RUN: vx-opt %s --vortex-lower-linalg-inside-kernel | FileCheck %s

func.func @fill_and_generic(
    %arg0: memref<4xf32, #vortex.address_space<global>>,
    %arg1: memref<4xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.500000e+00 : f32

  vortex.launch %c1, %c1, %c1 {
    linalg.fill ins(%cst : f32) outs(%arg1 : memref<4xf32, #vortex.address_space<global>>)
    linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%arg0 : memref<4xf32, #vortex.address_space<global>>)
        outs(%arg1 : memref<4xf32, #vortex.address_space<global>>) {
      ^bb0(%in: f32, %out: f32):
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
    }
    vortex.yield
  }
  return
}

func.func @matmul(
    %lhs: memref<2x2xf32, #vortex.address_space<global>>,
    %rhs: memref<2x2xf32, #vortex.address_space<global>>,
    %out: memref<2x2xf32, #vortex.address_space<global>>) attributes {vortex.kernel} {
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32

  vortex.launch %c1, %c1, %c1 {
    linalg.fill ins(%zero : f32) outs(%out : memref<2x2xf32, #vortex.address_space<global>>)
    linalg.matmul
      ins(%lhs, %rhs : memref<2x2xf32, #vortex.address_space<global>>, memref<2x2xf32, #vortex.address_space<global>>)
      outs(%out : memref<2x2xf32, #vortex.address_space<global>>)
    vortex.yield
  }
  return
}

// CHECK-LABEL: func.func @fill_and_generic(
// CHECK: vortex.launch %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK: scf.for %[[I:.*]] =
// CHECK: memref.store %cst, %{{.*}}[%{{.*}}] : memref<4xf32, #vortex.address_space<global>>
// CHECK: scf.for %[[J:.*]] =
// CHECK: %[[IN:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<4xf32, #vortex.address_space<global>>
// CHECK: %[[OUT:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<4xf32, #vortex.address_space<global>>
// CHECK: %[[SUM:.*]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK: memref.store %[[SUM]], %{{.*}}[%{{.*}}] : memref<4xf32, #vortex.address_space<global>>
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.generic

// CHECK-LABEL: func.func @matmul(
// CHECK: vortex.launch %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK: scf.for %[[M:.*]] =
// CHECK: scf.for %[[N:.*]] =
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf32, #vortex.address_space<global>>
// CHECK: scf.for %[[I2:.*]] =
// CHECK: scf.for %[[J2:.*]] =
// CHECK: scf.for %[[K2:.*]] =
// CHECK: %[[A:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf32, #vortex.address_space<global>>
// CHECK: %[[B:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf32, #vortex.address_space<global>>
// CHECK: %[[C:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf32, #vortex.address_space<global>>
// CHECK: %[[PROD:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK: %[[ACC:.*]] = arith.addf %[[C]], %[[PROD]] : f32
// CHECK: memref.store %[[ACC]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x2xf32, #vortex.address_space<global>>
// CHECK-NOT: linalg.matmul
