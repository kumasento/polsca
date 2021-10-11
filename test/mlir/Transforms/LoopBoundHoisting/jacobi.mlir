

// Jacobi style loops

#map1 = affine_map<(d0, d1)[s0] -> (0, (d0 * 32 - s0 + 1) ceildiv 2, d0 * 8 + d1 * 8)>
#map2 = affine_map<(d0, d1)[s0, s1] -> ((d0 * 32 + s0 + 29) floordiv 2 + 1, s1, d0 * 8 + d1 * 8 + 16)>
#map3 = affine_map<(d0, d1, d2) -> (d0 * 32, d1 * 32 + 2, d2 * 2 + 1, d1 * -32 + d2 * 4 - 31)>
#map4 = affine_map<(d0, d1, d2)[s0] -> (d0 * 32 + 32, d1 * -32 + d2 * 4 + 3, d1 * 32 + s0 * 2 + 28, d2 * 2 + s0)>

func @jacobi(%t: index, %n: index, %i: index, %j: index, %k: index, %A: memref<32x32xf32>) {
  affine.for %x = max #map1(%j, %i)[%n] to min #map2(%i, %j)[%n, %t] {
    affine.for %y = max #map3(%j, %i, %k) to min #map4(%j, %i, %k)[%n] {
      %0 = affine.load %A[%x, %y] : memref<32x32xf32>
      %1 = mulf %0, %0: f32
      affine.store %1, %A[%x, %y] : memref<32x32xf32>
    }
  }

  return
}
