// RUN: phism-opt %s -rewrite-ploop-indvar -canonicalize | FileCheck %s
#map0 = affine_map<(d0) -> (32 * d0)>
#map1 = affine_map<(d0) -> (32 * d0 + 32)>

func @foo() {
  %A = memref.alloca() : memref<1024xf32>
  %cst = arith.constant 1.0 : f32

  affine.for %i = 0 to 10 {
    affine.for %j = #map0(%i) to #map1(%i) {
      affine.store %cst, %A[%j] : memref<1024xf32>
    } {phism.point_loop}
  }
  return
}

// CHECK: func @foo
// CHECK: affine.for %[[i:.*]] = 0 to 10
// CHECK: affine.for %[[j:.*]] = 0 to 32
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[i]] * 32 + %[[j]]]
