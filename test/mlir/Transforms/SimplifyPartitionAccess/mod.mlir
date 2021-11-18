// RUN: phism-opt %s -simplify-partition-access | FileCheck %s
func @foo() {
  %A = memref.alloca() : memref<10xf32>
  %cst = arith.constant 1.0 : f32
  affine.for %i = 0 to 10 {
    affine.store %cst, %A[%i mod 10] : memref<10xf32>
  }

  return
}

// CHECK: func @foo
// CHECK: affine.for %[[i:.*]] = 
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[i]]]
