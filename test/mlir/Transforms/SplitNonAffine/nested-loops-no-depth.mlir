// RUN: phism-opt %s -split-non-affine | FileCheck %s

func @foo(%A: memref<?x?xf32>) {
  %cst = arith.constant 1.23 : f32
  %c1 = arith.constant 1: index
  affine.for %i = 0 to 10 {
    %j = arith.addi %i, %c1 : index
    memref.store %cst, %A[%i, %j] : memref<?x?xf32>
  }
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      affine.store %cst, %A[%i, %j] : memref<?x?xf32> 
    }
  }
  return
}


// CHECK: func @foo__f0
// CHECK-NEXT: affine.for
// CHECK-NEXT: affine.for
// CHECK-NEXT: affine.store

// CHECK: func @foo
// CHECK: affine.for
// CHECK: scop.non_affine_access
// CHECK: call @foo__f0
