; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -mem2arr -instcombine -S | FileCheck %s 

%memref2d = type { float*, float*, i64, [2 x i64], [2 x i64] }
%memref3d = type { float*, float*, i64, [3 x i64], [3 x i64] }

; CHECK: @arraynd_mixed([3 x [4 x float]]* %[[X:.*]], [3 x [4 x [5 x float]]]* %[[Y:.*]])
define float @arraynd_mixed(float* %a, float* %b) {
  %a2 = insertvalue %memref2d undef, float* %a, 0
  %a3 = insertvalue %memref2d %a2, float* %a, 1
  %a4 = insertvalue %memref2d %a3, i64 0, 2
  %a5 = insertvalue %memref2d %a4, i64 3, 3, 0
  %a6 = insertvalue %memref2d %a5, i64 4, 3, 1
  %a7 = insertvalue %memref2d %a6, i64 4, 4, 0
  %a8 = insertvalue %memref2d %a7, i64 1, 4, 1
  %a9 = extractvalue %memref2d %a8, 1

  %b2 = insertvalue %memref3d undef, float* %b, 0
  %b3 = insertvalue %memref3d %b2, float* %b, 1
  %b4 = insertvalue %memref3d %b3, i64 0, 2
  %b5 = insertvalue %memref3d %b4, i64 3, 3, 0
  %b6 = insertvalue %memref3d %b5, i64 4, 3, 1
  %b7 = insertvalue %memref3d %b6, i64 5, 3, 2
  %b8 = insertvalue %memref3d %b7, i64 20, 4, 0
  %b9 = insertvalue %memref3d %b8, i64 5, 4, 1
  %b10 = insertvalue %memref3d %b9, i64 1, 4, 2
  %b11 = extractvalue %memref3d %b10, 1

  ; CHECK: %{{.*}} = getelementptr inbounds [3 x [4 x float]], [3 x [4 x float]]* %[[X]], i64 0, i64 1, i64 2
  ; CHECK: %{{.*}} = getelementptr inbounds [3 x [4 x [5 x float]]], [3 x [4 x [5 x float]]]* %[[Y]], i64 0, i64 1, i64 0, i64 3  
  %x = getelementptr float, float* %a9, i64 6
  %y = getelementptr float, float* %b11, i64 23
  %x1 = load float, float* %x
  %y1 = load float, float* %y
  %z = fadd float %x1, %y1

  ret float %z
}
