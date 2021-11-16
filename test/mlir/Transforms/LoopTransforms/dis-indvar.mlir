// RUN: phism-opt %s -loop-transforms | FileCheck %s
func private @S0(%i: index, %j: index) attributes {scop.stmt}
func @foo() {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      affine.for %k = 0 to 20 {
        call @S0(%i, %k) : (index, index) -> ()
      }
    }
  }
  return
}

// CHECK:  func @foo__PE0(%[[i:.*]]: index) attributes {phism.pe} 
// CHECK:    affine.for %[[j:.*]] = 0 to 20 {
// CHECK:      call @S0(%[[i]], %[[j]]) : (index, index) -> ()
// CHECK:    } {phism.point_loop}

// CHECK:  func @foo() 
// CHECK:    affine.for %[[i:.*]] = 0 to 10 {
// CHECK:      affine.for %[[j:.*]] = 0 to 20 {
// CHECK:        call @foo__PE0(%[[i]]) {phism.pe} : (index) -> ()
// CHECK:      }
// CHECK:    } {phism.point_loop}
