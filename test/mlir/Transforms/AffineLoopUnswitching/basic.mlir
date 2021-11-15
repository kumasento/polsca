// RUN: phism-opt %s -affine-loop-unswitching | FileCheck %s
func private @S0()
func private @S1()
func @foo() {
  affine.for %i = 0 to 10 {
    call @S0() : () -> () 
    affine.if affine_set<(d0) : (-d0 + 5 >= 0)>(%i) {
      call @S1() : () -> () 
    }
  }
  return
}

// CHECK: func @foo() 
// CHECK:   affine.for %{{.*}} = 0 to 5 
// CHECK:     call @S0() 
// CHECK:     call @S1() 
// CHECK:   affine.for %{{.*}} = 5 to 10 
// CHECK:     call @S0() 
