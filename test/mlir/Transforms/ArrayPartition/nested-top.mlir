// RUN: phism-opt %s -array-partition | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 32 + 32)>

func @foo(%A: memref<128xf32>) attributes {phism.top} {
  %cst = arith.constant 3.14159 : f32
  affine.for %i = 0 to 128 {
    affine.store %cst, %A[%i] : memref<128xf32>
  }

  call @bar(%A) : (memref<128xf32>) -> ()

  return
}

func @bar(%A: memref<128xf32>) {
  affine.for %i = 0 to 4 {
    call @baz(%A, %i) {scop.pe} : (memref<128xf32>, index) -> ()
  }
  return
}

func @baz(%A: memref<128xf32>, %i: index) attributes {scop.pe} {
  %cst = arith.constant 1.23456 : f32
  affine.for %j = #map0()[%i] to #map1()[%i] {
    affine.store %cst, %A[%j] : memref<128xf32>
  }
  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 32 + 32)>

// CHECK:  func @foo(%[[A:.*]]: memref<4x32xf32>) attributes {phism.top} 
// CHECK:    affine.for %[[i:.*]] = 0 to 128 
// CHECK:      affine.store %{{.*}}, %[[A]][%[[i]] floordiv 32, %[[i]] mod 32]
// CHECK:    call @bar(%[[A]]) : (memref<4x32xf32>) -> ()

// CHECK:  func @bar(%[[A:.*]]: memref<4x32xf32>) 
// CHECK:    affine.for %[[i:.*]] = 0 to 4 
// CHECK:      call @baz(%[[A]], %[[i]]) {scop.pe} : (memref<4x32xf32>, index) -> ()

// CHECK:  func @baz(%[[A:.*]]: memref<4x32xf32>, %[[i:.*]]: index) attributes {scop.pe} 
// CHECK:    affine.for %[[j:.*]] = #map0()[%[[i]]] to #map1()[%[[i]]]
// CHECK:      affine.store %{{.*}}, %[[A]][%[[j]] floordiv 32, %[[j]] mod 32] : memref<4x32xf32>
