// RUN: phism-opt %s -loop-bound-hoisting | FileCheck %s

func @foo(%A: memref<30x40xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c30 = arith.constant 30 : index
  scf.for %i = %c0 to %c30 step %c1 {
    %c0_0 = arith.constant 0 : index
    %c40 = arith.constant 40 : index
    %c1_1 = arith.constant 1 : index
    scf.for %j = %c0_0 to %c40 step %c1_1 {
      %0 = memref.load %A[%i, %j] : memref<30x40xf32>
      %1 = arith.mulf %0, %0: f32
      memref.store %1, %A[%i, %j] : memref<30x40xf32>
    }
  }
  return
}

// CHECK-LABEL: func @foo
// CHECK: %{{.*}} = arith.constant 0
// CHECK-NEXT: %{{.*}} = arith.constant 1
// CHECK-NEXT: %{{.*}} = arith.constant 30
// CHECK-NEXT: %{{.*}} = arith.constant 0
// CHECK-NEXT: %{{.*}} = arith.constant 40
// CHECK-NEXT: %{{.*}} = arith.constant 1
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
