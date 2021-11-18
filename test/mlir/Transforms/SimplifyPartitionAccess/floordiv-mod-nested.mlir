// RUN: phism-opt %s -simplify-partition-access | FileCheck %s
func @foo() {
  %A = memref.alloca() : memref<10x2048xf32>
  %cst = arith.constant 1.0 : f32

  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      affine.store %cst, %A[(%i * 16 + %j) floordiv 256, (%i * 16 + %j) mod 256] : memref<10x2048xf32>
    }
  }

  return
}

// CHECK: func @foo
// CHECK: affine.for %[[i:.*]] = 0 to 16
// CHECK: affine.for %[[j:.*]] = 0 to 16
// CHECK: affine.store %{{.*}}, %{{.*}}[0, %[[i]] * 16 + %[[j]]]
