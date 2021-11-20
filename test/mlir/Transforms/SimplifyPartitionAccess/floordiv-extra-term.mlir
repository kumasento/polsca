// RUN: exit 0
func @foo(%A: memref<?xf32>) {
  %cst = arith.constant 1.0 : f32
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      affine.for %k = 1 to 20 {
        affine.store %cst, %A[(10 * %i + %k + %j * 256 - 1) floordiv 256] : memref<?xf32>
      }
    }
  }
  return
}
