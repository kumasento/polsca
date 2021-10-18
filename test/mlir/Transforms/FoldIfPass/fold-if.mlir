// RUN: phism-opt -fold-if %s | FileCheck %s

#set = affine_set<(d0) : (d0 - 5 >= 0)>

// CHECK-LABEL: func @square_last_half
func @square_last_half(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    affine.if #set(%i) {
      %0 = affine.load %A[%i] : memref<10xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %A[%i] : memref<10xf32>
    }  
  }
  return
}

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C5:.*]] = arith.constant -5 : index
// CHECK: %[[VAL0:.*]] = arith.addi %{{.*}}, %[[C5]] : index
// CHECK: %[[VAL1:.*]] = arith.cmpi sge, %[[VAL0]], %[[C0]] : index
// CHECK: %[[VAL2:.*]] = affine.load 
// CHECK: %[[VAL3:.*]] = arith.mulf %[[VAL2]], %[[VAL2]] : f32
// CHECK: %[[VAL4:.*]] = affine.load %[[ARG0:.*]][%[[ARG1:.*]]] : memref<10xf32>
// CHECK: %[[VAL5:.*]] = select %[[VAL1]], %[[VAL3]], %[[VAL4]] : f32
// CHECK: affine.store %[[VAL5]], %[[ARG0]][%[[ARG1]]] : memref<10xf32>
