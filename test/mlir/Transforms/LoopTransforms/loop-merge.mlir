// RUN: phism-opt -loop-merge %s | FileCheck %s

func @S0(%A: memref<32xf32>, %i: index) attributes {scop.stmt} {
  %0 = affine.load %A[%i] : memref<32xf32>
  %1 = arith.addf %0, %0 : f32
  affine.store %1, %A[%i] : memref<32xf32>
  return
}

func @two_loops(%A: memref<32xf32>) {
  affine.for %i = 0 to 16 {
    call @S0(%A, %i) : (memref<32xf32>, index) -> ()
  }
  affine.for %i = 16 to 32 {
    call @S0(%A, %i) : (memref<32xf32>, index) -> ()
  }
  return
}

// CHECK: func @two_loops
// CHECK: affine.for %[[ARG0:.*]] = 0 to 32
// CHECK: call @S0(%{{.*}}, %[[ARG0]])

func @top(%A : memref<32xf32>) {
  call @two_loops(%A) {scop.pe} : (memref<32xf32>) -> ()
  return 
}
