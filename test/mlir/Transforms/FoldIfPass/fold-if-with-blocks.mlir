// RUN: phism-opt %s -fold-if | FileCheck %s

#set = affine_set<(d0) : (d0 - 5 == 0)>

func @foo(%A: memref<?xf32>, %i: index, %a: f32) {
  affine.if #set(%i) {
    affine.store %a, %A[%i] : memref<?xf32>
    affine.for %j = 9 to 10 {
      affine.store %a, %A[%j] : memref<?xf32>
    }
  }
  return
}


// CHECK: func @foo
// CHECK: affine.load
// CHECK-NEXT: %[[v0:.*]] = select
// CHECK-NEXT: affine.store %[[v0]]
// CHECK: affine.for
// CHECK: affine.load
// CHECK-NEXT: %[[v0:.*]] = select
// CHECK-NEXT: affine.store %[[v0]]
