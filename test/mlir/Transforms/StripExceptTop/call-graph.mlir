// RUN: phism-opt %s -strip-except-top | FileCheck %s

func @foo() attributes {phism.top} {
  call @bar(): () -> ()
  return
}
func @bar() { return }
func @baz() { return }

// CHECK: func @foo
// CHECK: func @bar
// CHECK-NOT: func @baz
