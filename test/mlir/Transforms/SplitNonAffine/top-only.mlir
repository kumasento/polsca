// RUN: phism-opt %s -split-non-affine='top-only=1' | FileCheck %s
func @foo() attributes {phism.top} {
  affine.for %i = 0 to 10 {}
  return
}
func @bar() {
  affine.for %i = 0 to 10 {}
  return
}

// CHECK: func @foo
// CHECK-NEXT: call @foo__f0

// CHECK: func @bar
// CHECK-NEXT: affine.for
