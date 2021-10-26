// RUN: phism-opt %s -eliminate-affine-load-store | FileCheck %s 

func @foo(%A: memref<?xf32>, %i: index, %a: f32) -> f32 {
  affine.store %a, %A[%i] : memref<?xf32>
  %v = affine.load %A[%i] : memref<?xf32>
  affine.store %a, %A[%i] : memref<?xf32>
  return %v : f32
}

// CHECK: func @foo(%[[A:.*]]: memref<?xf32>, %[[i:.*]]: index, %[[a:.*]]: f32)
// CHECK-NEXT: affine.store %[[a]], %[[A]][%[[i]]]
// CHECK-NEXT: return %[[a]]
