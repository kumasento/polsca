// RUN: phism-opt %s -extract-top-func='name=foo' | FileCheck %s

func @foo() {
  return 
}
func @bar() {
  return
}

// CHECK-LABEL: func @foo
// CHECK-NOT: func @bar
