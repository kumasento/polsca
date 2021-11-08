// RUN: phism-opt %s -simplify-partition-access -canonicalize | FileCheck %s

#map0 = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 32 + 32)>

func @foo(%A: memref<4x32xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = #map0(%i) to #map1(%i) {
      %0 = affine.load %A[%j floordiv 32, %j mod 32] : memref<4x32xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %A[%j floordiv 32, %j mod 32] : memref<4x32xf32>
    }
  }

  return
}

// CHECK: func @foo(%[[A:.*]]: memref<4x32xf32>)
// CHECK: affine.for %[[i:.*]] = 0 to 4
// CHECK: affine.for %[[j:.*]] = #{{.*}}(%[[i]]) to #{{.*}}(%[[i]])
// CHECK: %{{.*}} = affine.load %[[A]][%[[i]], %[[j]] mod 32]
// CHECK: affine.store %{{.*}}, %[[A]][%[[i]], %[[j]] mod 32]
