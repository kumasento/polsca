// RUN: phism-opt %s -extract-top-func='name=foo' | FileCheck %s

func @foo() {
  call @bar(): () -> ()
  call @baz(): () -> ()
  return
}
func @bar() {
  call @qux(): () -> ()
  return
}
func @baz() {
  return
}
func @qux() {
  call @foo() : () -> ()
  return
}
func @quux() {
  return
}

// CHECK: func @foo
// CHECK: func @bar
// CHECK: func @baz
// CHECK: func @qux
// CHECK-NOT: func @quux
