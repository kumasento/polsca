// RUN: phism-opt %s -simplify-partition-access | FileCheck %s
func @foo(%A: memref<?xf32>) {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 32 {
      %0 = affine.load %A[%i + %j floordiv 32] : memref<?xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %A[%i + %j floordiv 32] : memref<?xf32>
    }
  }

  return
}

// CHECK: func @foo
// CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}}]
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}]
