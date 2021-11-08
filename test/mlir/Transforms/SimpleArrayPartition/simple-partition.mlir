// RUN: phism-opt -simple-array-partition -canonicalize %s | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 32 + 32)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK: func @bar(%[[ARG0:.*]]: memref<32xf32>, %[[ARG1:.*]]: index)
func @bar(%A: memref<64xf32>, %i: index) {
  // CHECK: affine.for %[[ARG2:.*]] =  
  affine.for %j = #map0()[%i] to #map1()[%i] {
    // CHECK: affine.load %[[ARG0]][%[[ARG2]] mod 32]
    %0 = affine.load %A[%j] : memref<64xf32>
    %1 = arith.addf %0, %0 : f32
    // CHECK: affine.store %{{.*}}, %[[ARG0]][%[[ARG2]] mod 32]
    affine.store %1, %A[%j] : memref<64xf32>
  }
  return
}

// CHECK: func @foo(%[[ARG0:.*]]: memref<2x32xf32>)
func @foo(%A: memref<64xf32>) attributes {phism.top} {
  // CHECK: affine.for %[[ARG1:.*]] = 0 to 2 
  affine.for %i = 0 to 2 {
    // CHECK: %[[VAL0:.*]] = memref.subview %[[ARG0]][%[[ARG1]], 0] [1, 32] [1, 1] : memref<2x32xf32> to memref<32xf32, #[[MAP2]]>
    // CHECK-NEXT: %[[VAL1:.*]] = memref.cast %[[VAL0]] : memref<32xf32, #[[MAP2]]> to memref<32xf32>
    // CHECK-NEXT: call @bar(%[[VAL1]], %[[ARG1]]) {scop.pe} : (memref<32xf32>, index) -> ()
    call @bar(%A, %i) {scop.pe} : (memref<64xf32>, index) -> ()
  }
  return
}
