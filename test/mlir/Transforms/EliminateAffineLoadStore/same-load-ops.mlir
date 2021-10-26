// RUN: phism-opt %s -eliminate-affine-load-store | FileCheck %s

func @foo(%A: memref<?xi32>) {
  %fst = affine.load %A[0] : memref<?xi32>
  %fst2 = affine.load %A[0] : memref<?xi32>
  // Should only leave one.

  affine.for %i = 1 to 10 {
    %cur1 = affine.load %A[%i] : memref<?xi32>
    %cur2 = affine.load %A[%i] : memref<?xi32>
    %next = affine.load %A[%i + 1] : memref<?xi32> // should not remove

    %doubled = arith.addi %cur1, %cur2 : i32
    %sum = arith.addi %next, %doubled : i32
    %final = arith.addi %fst2, %sum : i32

    affine.store %final, %A[100 + %i] : memref<?xi32>
  }

  return
}

// CHECK: func @foo(%[[A:.*]]: memref<?xi32>)
// CHECK-NEXT: %[[v0:.*]] = affine.load %[[A]][0]
// CHECK-NEXT: affine.for %[[i:.*]] = 1 to 10
// CHECK-NEXT: %[[v1:.*]] = affine.load %[[A]][%[[i]]]
// CHECK-NEXT: %[[v2:.*]] = affine.load %[[A]][%[[i]] + 1]
// CHECK-NEXT: %[[v3:.*]] = arith.addi %[[v1]], %[[v1]]
// CHECK-NEXT: %[[v4:.*]] = arith.addi %[[v2]], %[[v3]]
// CHECK-NEXT: %[[v5:.*]] = arith.addi %[[v0]], %[[v4]]
// CHECK-NEXT: affine.store %[[v5]], %[[A]][%[[i]] + 100]
