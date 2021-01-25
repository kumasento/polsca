// RUN: phism-opt %s -convert-std-to-llvm -mem2ptr-call

func @foo(%A: memref<?xf32>) {
  %c0 = constant 0 : index
  %cst = constant 1.23 : f32
  store %cst, %A[%c0] : memref<?xf32>

  return 
}
