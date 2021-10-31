// RUN: phism-opt %s -split-non-affine='max-loop-depth=1' | FileCheck %s --check-prefix DEPTH
// RUN: phism-opt %s -split-non-affine='max-loop-depth=0' | FileCheck %s 

func @foo(%A: memref<?x?xf32>) {
  %cst = arith.constant 1.23 : f32
  %c1 = arith.constant 1: index
  affine.for %i = 0 to 10 {
    %j = arith.addi %i, %c1 : index
    memref.store %cst, %A[%i, %j] : memref<?x?xf32>
    affine.for %k = 0 to 20 {
      affine.store %cst, %A[%i, %k] : memref<?x?xf32> 
    }
  }
  return
}


// DEPTH: func @foo__f0
// DEPTH-NEXT: affine.for
// DEPTH-NEXT: affine.store

// DEPTH: func @foo
// DEPTH: affine.for
// DEPTH-NEXT: addi
// DEPTH-NEXT: memref.store
// DEPTH-NEXT: call @foo__f0
// DEPTH-NEXT: scop.non_affine_access


// CHECK: func @foo
// CHECK: affine.for
// CHECK-NEXT: addi
// CHECK-NEXT: memref.store
// CHECK-NEXT: affine.for
// CHECK-NEXT: affine.store
// CHECK: scop.non_affine_access
