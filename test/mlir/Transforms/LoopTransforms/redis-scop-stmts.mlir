// RUN: phism-opt %s -redis-scop-stmts | FileCheck %s

func @S0(%A: memref<32xf32>, %i: index) attributes {scop.stmt} {
  %0 = affine.load %A[%i] : memref<32xf32>
  %1 = arith.addf %0, %0 : f32
  affine.store %1, %A[%i] : memref<32xf32>
  return
}

func @S1(%A: memref<32xf32>, %i: index) attributes {scop.stmt} {
  %0 = affine.load %A[%i] : memref<32xf32>
  %1 = arith.mulf %0, %0 : f32
  affine.store %1, %A[%i] : memref<32xf32>
  return
}

func @two_stmts(%A: memref<32xf32>, %B: memref<32xf32>) {
  affine.for %i = 0 to 32 {
    call @S0(%A, %i) : (memref<32xf32>, index) -> ()
    call @S1(%B, %i) : (memref<32xf32>, index) -> ()
  }
  return
}

// CHECK: func @two_stmts__cloned_for__S1(%[[ARG0:.*]]: memref<32xf32>)
// CHECK: affine.for %[[ARG1:.*]] = 0 to 32 
// CHECK: call @S1(%[[ARG0]], %[[ARG1]])

// CHECK: func @two_stmts__cloned_for__S0(%[[ARG0:.*]]: memref<32xf32>)
// CHECK: affine.for %[[ARG1:.*]] = 0 to 32 
// CHECK: call @S0(%[[ARG0]], %[[ARG1]])

// CHECK: func @top(%[[ARG0:.*]]: memref<32xf32>, %[[ARG1:.*]]: memref<32xf32>)
// CHECK: call @two_stmts__cloned_for__S0(%[[ARG0]])
// CHECK: call @two_stmts__cloned_for__S1(%[[ARG1]])

func @top(%A : memref<32xf32>, %B : memref<32xf32>) attributes {phism.top} {
  call @two_stmts(%A, %B) {phism.pe} : (memref<32xf32>, memref<32xf32>) -> ()
  return 
}
