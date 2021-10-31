// RUN: phism-opt %s -extract-top-func='name=foo' | FileCheck %s

func @foo() {
  return
}

func @main() {
  call @foo() : () -> ()
  return
}

// CHECK: func @foo() attributes {phism.top}
