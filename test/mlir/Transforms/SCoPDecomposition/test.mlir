// RUN: phism-opt %s -scop-decomp | FileCheck %s
func @foo(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>, %D: memref<?xf32>) attributes {phism.top} {
  %cst = arith.constant 1.0 : f32
  affine.for %t = 0 to 20 {
    affine.for %i = 0 to 10 {
      affine.store %cst, %A[%i] : memref<?xf32>
      affine.store %cst, %C[%i] : memref<?xf32>
    }
    affine.for %i = 0 to 20 {
      %0 = affine.load %A[%i] : memref<?xf32>
      affine.store %0, %B[%i] : memref<?xf32>
    }
    affine.for %i = 0 to 30 {
      %0 = affine.load %C[%i] : memref<?xf32>
      affine.store %0, %D[%i] : memref<?xf32>
    }
    affine.for %i = 0 to 50 {
      affine.store %cst, %B[%i] : memref<?xf32>
      affine.store %cst, %D[%i] : memref<?xf32>
    }
  }
  return
}

// CHECK:  func @foo__f0(%{{.*}}) attributes {scop.affine} 
// CHECK:    affine.for %{{.*}} = 0 to 10 
// CHECK:  func @foo__f1(%{{.*}}) attributes {scop.affine} 
// CHECK:    affine.for %{{.*}} = 0 to 20 
// CHECK:    affine.for %{{.*}} = 0 to 30 
// CHECK:  func @foo__f2(%{{.*}}) attributes {scop.affine} 
// CHECK:    affine.for %{{.*}} = 0 to 50 
// CHECK:  func @foo(%{{.*}}) attributes {phism.top, scop.ignored} 
// CHECK:    affine.for %{{.*}} = 0 to 20 
// CHECK:      call @foo__f0(%{{.*}}) {scop.affine}
// CHECK:      call @foo__f1(%{{.*}}) {scop.affine}
// CHECK:      call @foo__f2(%{{.*}}) {scop.affine}
