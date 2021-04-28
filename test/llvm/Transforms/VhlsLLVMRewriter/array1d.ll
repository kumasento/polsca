; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -mem2arr -S | FileCheck %s 

%memref = type { float*, float*, i64, [1 x i64], [1 x i64] }

; CHECK-LABEL: @array1d([4 x float]* %0)
define void @array1d(float* %0) {
  %2 = insertvalue %memref undef, float* %0, 0
  %3 = insertvalue %memref %2, float* %0, 1
  %4 = insertvalue %memref %3, i64 0, 2
  %5 = insertvalue %memref %4, i64 4, 3, 0
  %6 = insertvalue %memref %5, i64 1, 4, 0
  %7 = extractvalue %memref %6, 1

  ; CHECK-NEXT: %{{.*}} = getelementptr inbounds [4 x float], [4 x float]* %0, i64 0, i64 1
  %8 = getelementptr float, float* %7, i64 1
  ; CHECK-NEXT: ret void
  ret void
}
