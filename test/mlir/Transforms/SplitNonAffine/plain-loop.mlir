// RUN: phism-opt %s -split-non-affine='mark-only=1' | FileCheck %s --check-prefix MARK
// RUN: phism-opt %s -split-non-affine | FileCheck %s

func @foo(%A: memref<?xf32>, %B: memref<?xi32>) {
  affine.for %i = 0 to 10 {
    %0 = affine.load %B[%i] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %A[%1] : memref<?xf32>
    affine.store %2, %A[%i] : memref<?xf32>
  }

  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.store %0, %A[%i] : memref<?xf32>
  }

  return
}

// MARK: func @foo
// MARK: affine.for
// MARK: scop.non_affine_access
// MARK-NEXT: affine.for

// CHECK: func @foo__f0
// CHECK-NEXT: affine.for
// CHECK-NEXT: affine.load
// CHECK-NEXT: affine.store

// CHECK: func @foo
// CHECK: affine.for
// CHECK: scop.non_affine_access
// CHECK-NEXT: call @foo__f0
