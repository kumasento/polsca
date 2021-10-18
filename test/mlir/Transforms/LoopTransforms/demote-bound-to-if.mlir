// RUN: phism-opt %s -demote-bound-to-if | FileCheck %s


// Jacobi style loops

#map0 = affine_map<()[s0, s1, s2] -> (0, (-s0 + s1 * 32 + 1) ceildiv 2, s1 * 8 + s2 * 8)>
#map1 = affine_map<()[s0, s1, s2, s3] -> ((s0 + s2 * 32 + 29) floordiv 2 + 1, s1, s2 * 8 + s3 * 8 + 16)>
#map2 = affine_map<(d0)[s0, s1] -> (s0 * 32, s1 * 32 + 2, d0 * 2 + 1, d0 * 4 - s1 * 32 - 31)>
#map3 = affine_map<(d0)[s0, s1, s2] -> (s1 * 32 + 32, d0 * 4 - s2 * 32 + 3, s0 * 2 + s2 * 32 + 28, d0 * 2 + s0)>

func @jacobi(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<32x32xf32>) attributes {scop.pe} {
  affine.for %arg5 = max #map0()[%arg1, %arg3, %arg2] to min #map1()[%arg1, %arg0, %arg2, %arg3] {
    affine.for %arg6 = max #map2(%arg5)[%arg3, %arg2] to min #map3(%arg5)[%arg1, %arg3, %arg2] {
      %0 = affine.load %arg4[%arg5, %arg6] : memref<32x32xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg4[%arg5, %arg6] : memref<32x32xf32>
    }
  }
  return
}


// CHECK-LABEL: func @jacobi
// CHECK: affine.for 
// CHECK: affine.for %{{.*}} = max #{{.*}}()[%{{.*}}, %{{.*}}] to min #{{.*}}()[%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK: affine.if 
