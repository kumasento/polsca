; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -mem2arr -S | FileCheck %s 

%memref = type { float*, float*, i64, [1 x i64], [1 x i64] }

; CHECK-LABEL: @array1d_multiargs([4 x float]* %0, [8 x float]* %1)
define void @array1d_multiargs(float* %a, float* %b) {
  %a.2 = insertvalue %memref undef, float* %a, 0
  %a.3 = insertvalue %memref %a.2, float* %a, 1
  %a.4 = insertvalue %memref %a.3, i64 0, 2
  %a.5 = insertvalue %memref %a.4, i64 4, 3, 0
  %a.6 = insertvalue %memref %a.5, i64 1, 4, 0

  %b.2 = insertvalue %memref undef, float* %b, 0
  %b.3 = insertvalue %memref %b.2, float* %b, 1
  %b.4 = insertvalue %memref %b.3, i64 0, 2
  %b.5 = insertvalue %memref %b.4, i64 8, 3, 0
  %b.6 = insertvalue %memref %b.5, i64 1, 4, 0

  %a.7 = extractvalue %memref %a.6, 1
  %b.7 = extractvalue %memref %b.6, 1

  ; CHECK-NEXT: %{{.*}} = getelementptr inbounds [4 x float], [4 x float]* %0, i64 0, i64 1
  ; CHECK-NEXT: %{{.*}} = getelementptr inbounds [8 x float], [8 x float]* %1, i64 0, i64 2
  %a.ptr = getelementptr float, float* %a.7, i64 1
  %b.ptr = getelementptr float, float* %b.7, i64 2

  ret void
}
