// RUN: phism-opt %s -extract-top-func='name=foo' | FileCheck %s

func @foo(%a: index) {
  return
}

func @main() {
  %0 = arith.constant 10 : index
  call @foo(%0) : (index) -> ()
  return
}

// CHECK: func @foo(%{{.*}}: index {scop.constant_value = 10 : index})
