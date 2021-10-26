// RUN: phism-opt %s -eliminate-affine-load-store='load-after-store=0' | FileCheck %s

// All the loads won't be eliminated.
func @foo(%A: memref<?xi32>, %i: index, %a: i32) -> i32 {
  %fst = affine.load %A[%i] : memref<?xi32>
  affine.store %a, %A[%i] : memref<?xi32>
  %snd = affine.load %A[%i] : memref<?xi32>
  affine.for %t = 0 to 2 {}
  %trd = affine.load %A[%i] : memref<?xi32>
  return %snd : i32
}

// CHECK: func @foo
// CHECK-NEXT: affine.load
// CHECK-NEXT: affine.store
// CHECK-NEXT: affine.load
// CHECK-NEXT: affine.for
// CHECK-NEXT: }
// CHECK-NEXT: affine.load
// CHECK-NEXT: return
