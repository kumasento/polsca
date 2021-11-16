// RUN: phism-opt %s -anno-point-loop | FileCheck %s
func private @S0(%i: index, %j: index) attributes {scop.stmt}
func @foo(%N: index) {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to %N {
      call @S0(%i, %j): (index, index) -> ()
    }
  }
  return
}

// CHECK: func @foo
// CHECK-NEXT: affine.for 
// CHECK-NEXT: affine.for 
// CHECK-NEXT: call @S0
// CHECK-NEXT: } {phism.point_loop}
// CHECK-NEXT: }
// CHECK-NEXT: return
