// RUN: phism-opt -simple-array-partition %s | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 32 + 32)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>

// CHECK: func @bar(%[[ARG0:.*]]: memref<32x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func @bar(%A: memref<64x96xf32>, %i: index, %j: index) {
  // CHECK: affine.for %[[ARG3:.*]] = 
  affine.for %k = #map0()[%i] to #map1()[%i] {
    // CHECK: affine.for %[[ARG4:.*]] = 
    affine.for %l = #map0()[%j] to #map1()[%j] {
      // CHECK: affine.load %[[ARG0]][%[[ARG3]] mod 32, %[[ARG4]] mod 32] 
      %0 = affine.load %A[%k, %l] : memref<64x96xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %A[%k, %l] : memref<64x96xf32>
    }
  }
  return
}

// CHECK: func @foo(%[[A:.*]]: memref<2x3x32x32xf32>)
func @foo(%A: memref<64x96xf32>) {
  // CHECK: affine.for %[[ARG1:.*]] = 0 to 2
  affine.for %i = 0 to 2 {
    // CHECK: affine.for %[[ARG2:.*]] = 0 to 3
    affine.for %j = 0 to 3 {
      // CHECK: %[[VAL0:.*]] = memref.subview %[[A]][%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x3x32x32xf32> to memref<32x32xf32, #[[MAP2]]>
      // CHECK-NEXT: %[[VAL1:.*]] = memref.cast %[[VAL0]] : memref<32x32xf32, #[[MAP2]]> to memref<32x32xf32>
      // CHECK-NEXT: call @bar(%[[VAL1]], %[[ARG1]], %[[ARG2]])
      call @bar(%A, %i, %j) {scop.pe} : (memref<64x96xf32>, index, index) -> ()
    }
  }
  return
}
