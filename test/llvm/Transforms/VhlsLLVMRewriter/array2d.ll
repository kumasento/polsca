; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -mem2arr -instcombine -S | FileCheck %s 

%memref = type { float*, float*, i64, [2 x i64], [2 x i64] }

; CHECK: @array2d([3 x [4 x float]]* %[[ARR:.*]])
define float @array2d(float* %0) {
  %2 = insertvalue %memref undef, float* %0, 0
  %3 = insertvalue %memref %2, float* %0, 1
  %4 = insertvalue %memref %3, i64 0, 2
  %5 = insertvalue %memref %4, i64 3, 3, 0
  %6 = insertvalue %memref %5, i64 4, 3, 1
  %7 = insertvalue %memref %6, i64 4, 4, 0
  %8 = insertvalue %memref %7, i64 1, 4, 1
  %9 = extractvalue %memref %8, 1

  ; CHECK-NEXT: %[[PTR:.*]] = getelementptr inbounds [3 x [4 x float]], [3 x [4 x float]]* %[[ARR]], i64 0, i64 1, i64 2
  %10 = getelementptr float, float* %9, i64 6
  ; CHECK-NEXT: %[[VAL:.*]] = load float, float* %[[PTR]]
  %11 = load float, float* %10
  ; CHECK-NEXT: ret float %[[VAL]]
  ret float %11
}

