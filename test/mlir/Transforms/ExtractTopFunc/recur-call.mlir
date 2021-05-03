// RUN: phism-opt %s -extract-top-func='name=foo' | FileCheck %s

func @foo() {
  call @foo() : () -> ()
  return
}

// CHECK-COUNT-1: func @foo
