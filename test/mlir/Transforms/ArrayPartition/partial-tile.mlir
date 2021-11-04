// RUN: phism-opt %s -array-partition | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 32 + 32)>

func @bar(%A: memref<128x128xf32>, %i: index, %j: index) attributes {scop.pe} {
  %cst = arith.constant 1.23 : f32
  affine.for %k = #map0()[%j] to #map1()[%j] {
    affine.store %cst, %A[%i, %k] : memref<128x128xf32>
  }
  return
}
func @foo(%A: memref<128x128xf32>, %i : index) {
  affine.for %j = 0 to 1 {
    call @bar(%A, %i, %j) {scop.pe} : (memref<128x128xf32>, index, index) -> ()
  }
  return
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 32 + 32)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>

// CHECK:   func @bar(%[[A:.*]]: memref<128x32xf32>, %[[i:.*]]: index, %[[j:.*]]: index) 
// CHECK:     affine.for %[[k:.*]] = #[[MAP0]]()[%[[j]]] to #[[MAP1]]()[%[[j]]] {
// CHECK:       affine.store %{{.*}}, %[[A]][symbol(%[[i]]), %[[k]] mod 32] : memref<128x32xf32>

// CHECK:   func @foo(%[[A:.*]]: memref<1x4x128x32xf32>, %[[i:.*]]: index) 
// CHECK:     affine.for %[[j:.*]] = 0 to 1 
// CHECK:       %[[v0:.*]] = memref.subview %[[A]][0, %[[j]], 0, 0] [1, 1, 128, 32] [1, 1, 1, 1] : memref<1x4x128x32xf32> to memref<128x32xf32, #[[MAP2]]>
// CHECK:       %[[v1:.*]] = memref.cast %[[v0]] : memref<128x32xf32, #[[MAP2]]> to memref<128x32xf32>
// CHECK:       call @bar(%1, %arg1, %arg2) {scop.pe} : (memref<128x32xf32>, index, index) -> ()
