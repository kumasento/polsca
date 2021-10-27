// RUN: phism-opt %s -extract-top-func="name=foo keepall=1" | FileCheck %s

func private @bar() { return }
func @foo() { return }

// CHECK: func private @bar
// CHECK: func @foo
