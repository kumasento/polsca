// RUN: mlir-opt -lower-affine -convert-scf-to-std -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' %s | mlir-translate -mlir-to-llvmir | opt -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -mem2arr -instcombine -strip-debug -S | FileCheck %s 

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

// CHECK: %[[VAL14:.*]] = mul i64 %[[I:.*]], 300
// CHECK: %[[VAL15:.*]] = add i64 %[[VAL14]], %[[K:.*]]
// CHECK: %[[GEP0IDX0:.*]] = udiv i64 %[[VAL15]], 300
// CHECK: %[[GEP0ADDR1:.*]] = urem i64 %[[VAL15]], 300
// CHECK: %[[GEP0:.*]] = getelementptr inbounds [200 x [300 x float]], [200 x [300 x float]]* %[[A]], i64 0, i64 %[[GEP0IDX0]], i64 %[[GEP0ADDR1]]
// CHECK: %[[VAL16:.*]] = load float, float* %[[GEP0]], align 4
// CHECK: %[[VAL17:.*]] = mul i64 %[[K]], 400
// CHECK: %[[VAL18:.*]] = add i64 %[[VAL17]], %[[J:.*]]
// CHECK: %[[GEP1IDX0:.*]] = udiv i64 %[[VAL18]], 400
// CHECK: %[[GEP1ADDR1:.*]] = urem i64 %[[VAL18]], 400
// CHECK: %[[GEP1:.*]] = getelementptr inbounds [300 x [400 x float]], [300 x [400 x float]]* %[[B]], i64 0, i64 %[[GEP1IDX0]], i64 %[[GEP1ADDR1]]
// CHECK: %[[VAL19:.*]] = load float, float* %[[GEP1]], align 4
// CHECK: %[[VAL20:.*]] = fmul float %[[VAL16]], %[[VAL19]]
// CHECK: %[[VAL21:.*]] = mul i64 %[[I]], 400
// CHECK: %[[VAL22:.*]] = add i64 %[[VAL21]], %[[J]]
// CHECK: %[[GEP2IDX0:.*]] = udiv i64 %[[VAL22]], 400
// CHECK: %[[GEP2ADDR1:.*]] = urem i64 %[[VAL22]], 400
// CHECK: %[[GEP2:.*]] = getelementptr inbounds [200 x [400 x float]], [200 x [400 x float]]* %[[C]], i64 0, i64 %[[GEP2IDX0]], i64 %[[GEP2ADDR1]]
// CHECK: store float %[[VAL20]], float* %[[GEP2]], align 4
