// RUN: phism-opt -simple-array-partition="flatten" %s | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 32 + 32)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>

func @bar(%A: memref<64x96xf32>, %i: index, %j: index) {
  affine.for %k = #map0()[%i] to #map1()[%i] {
    affine.for %l = #map0()[%j] to #map1()[%j] {
      %0 = affine.load %A[%k, %l] : memref<64x96xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %A[%k, %l] : memref<64x96xf32>
    }
  }
  return
}


// CHECK: func @foo(%[[ARG0:.*]]: memref<6x32x32xf32>)
func @foo(%A: memref<64x96xf32>) attributes {phism.top} {
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {

      // CHECK: affine.for %[[ARG1:.*]] = 0 to 2 
      // CHECK: affine.for %[[ARG2:.*]] = 0 to 3 
      // CHECK: %[[VAL0:.*]] = affine.apply #[[MAP2]](%[[ARG1]], %[[ARG2]])
      // CHECK: %[[VAL1:.*]] = memref.subview %[[ARG0]][%[[VAL0]], 0, 0] [1, 32, 32] [1, 1, 1]

      call @bar(%A, %i, %j) {phism.pe} : (memref<64x96xf32>, index, index) -> ()
    }
  }
  return
}
