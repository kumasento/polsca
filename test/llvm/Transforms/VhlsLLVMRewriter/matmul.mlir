// RUN: mlir-opt -lower-affine -convert-scf-to-std -convert-memref-to-llvm -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' %s | mlir-translate -mlir-to-llvmir | opt -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -mem2arr -instcombine -strip-debug -S | FileCheck %s 

// CHECK: noinline
// CHECK: define void @matmul([200 x [300 x float]]* %[[A:.*]], [300 x [400 x float]]* %[[B:.*]], [200 x [400 x float]]* %[[C:.*]]) #[[ATTR:.*]]
func @matmul(%A: memref<200x300xf32>, %B: memref<300x400xf32>, %C: memref<200x400xf32>) {
  affine.for %i = 0 to 200 {
    affine.for %j = 0 to 400 {
      %0 = affine.load %C[%i, %j] : memref<200x400xf32>
      affine.for %k = 0 to 300 {
        %1 = affine.load %A[%i, %k] : memref<200x300xf32>
        %2 = affine.load %B[%k, %j] : memref<300x400xf32>
        %3 = mulf %1, %2 : f32
        affine.store %3, %C[%i, %j] : memref<200x400xf32>
      }
    }
  }
  return
}

// CHECK: %[[i:.*]] = phi i64
// CHECK: %[[j:.*]] = phi i64
// CHECK: %[[k:.*]] = phi i64
// CHECK: getelementptr inbounds [200 x [300 x float]], [200 x [300 x float]]* %[[A]], i64 0, i64 %[[i]], i64 %[[k]]
// CHECK: getelementptr inbounds [300 x [400 x float]], [300 x [400 x float]]* %[[B]], i64 0, i64 %[[k]], i64 %[[j]]
// CHECK: getelementptr inbounds [200 x [400 x float]], [200 x [400 x float]]* %[[C]], i64 0, i64 %[[i]], i64 %[[j]]
