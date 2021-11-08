// RUN: phism-opt %s -array-partition -canonicalize | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 32 + 32)>

func @bar(%A: memref<128x128xf32>, %i: index, %j: index) attributes {scop.pe} {
  %cst = arith.constant 1.23 : f32
  affine.for %k = #map0()[%j] to #map1()[%j] {
    affine.store %cst, %A[symbol(%i), %k] : memref<128x128xf32>
  }
  return
}
func @foo(%A: memref<128x128xf32>, %i : index) attributes {phism.top} {
  affine.for %j = 0 to 1 {
    call @bar(%A, %i, %j) {scop.pe} : (memref<128x128xf32>, index, index) -> ()
  }
  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 32 + 32)>

// CHECK: func @bar(%[[A:.*]]: memref<1x4x128x32xf32>, %[[i:.*]]: index, %[[j:.*]]: index)
// CHECK:   affine.for %[[k:.*]] = #[[MAP0]]()[%[[j]]] to #[[MAP1]]()[%[[j]]] 
// CHECK:     affine.store %{{.*}}, %[[A]][0, %[[k]] floordiv 32, symbol(%[[i]]), %[[k]] mod 32] : memref<1x4x128x32xf32>
// CHECK: func @foo(%[[A:.*]]: memref<1x4x128x32xf32>, %[[i:.*]]: index) attributes {phism.top} 
// CHECK:   affine.for %[[j:.*]] = 0 to 1 
// CHECK:     call @bar(%[[A]], %[[i]], %[[j]]) {scop.pe} : (memref<1x4x128x32xf32>, index, index) -> ()

