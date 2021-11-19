// RUN: phism-opt %s -lift-memref-subview | FileCheck %s

func @bar(%A: memref<10x20xf32>, %i: index) attributes {phism.pe} {
  %cst = arith.constant 1.0 : f32
  affine.for %j = 0 to 20 {
    affine.store %cst, %A[symbol(%i), %j] : memref<10x20xf32>
  }
  return
}

func @foo() {
  %A = memref.alloca() : memref<10x20xf32>
  affine.for %i = 0 to 10 {
    call @bar(%A, %i) {phism.pe} : (memref<10x20xf32>, index) -> () 
  }
  return
}


// CHECK: func @bar(%[[A:.*]]: memref<20xf32>, %[[i:.*]]: index)
// CHECK:   affine.for %[[j:.*]] = 0 to 20 {
// CHECK:     affine.store %{{.*}}, %[[A]][%[[j]]] : memref<20xf32>
// CHECK: func @foo() 
// CHECK:   affine.for %[[i:.*]] = 0 to 10 
// CHECK:     %[[v1:.*]] = memref.subview %{{.*}}[%[[i]], 0] [1, 20] [1, 1] : memref<10x20xf32> to memref<20xf32, #{{.*}}>
// CHECK:     %[[v2:.*]] = memref.cast %[[v1]] : memref<20xf32, #{{.*}}> to memref<20xf32>
// CHECK:     call @bar(%[[v2]], %[[i]]) {phism.pe} : (memref<20xf32>, index) -> ()
