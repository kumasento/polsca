// RUN: phism-opt -simple-array-partition %s  -verify-diagnostics | FileCheck %s

#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 32 + 32)>

// expected-remark@below {{No top function found}}
module {

// CHECK-LABEL: @bar
func @bar(%A: memref<64xf32>, %i: index) {
  affine.for %j = #map0()[%i] to #map1()[%i] {
    %0 = affine.load %A[%j] : memref<64xf32>
    %1 = addf %0, %0 : f32
    affine.store %1, %A[%j] : memref<64xf32>
  }
  return
}

// CHECK-LABEL: @foo
func @foo(%A: memref<64xf32>) {
  affine.for %i = 0 to 2 {
    call @bar(%A, %i) : (memref<64xf32>, index) -> ()
  }
  return
}

}
