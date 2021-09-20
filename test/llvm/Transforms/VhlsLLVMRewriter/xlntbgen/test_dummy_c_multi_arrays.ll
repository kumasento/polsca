; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -xlntbgen -xlntop="foo" -S -xlntbdummynames="%t" && cat %t | FileCheck %s 

define void @foo(i32 %n, [32 x float]* %A, [4 x [8 x i32]]* %B) {
  ret void
}

; CHECK: void foo(int n, float A[32], int B[4][8]) {
; CHECK:   A[n + 1] += A[n];
; CHECK:   B[n + 1][n + 1] += B[n][n];
; CHECK: }
; CHECK: int main() {
; CHECK:   int n;
; CHECK:   float A[32];
; CHECK:   int B[4][8];
; CHECK:   foo(n, A, B);
; CHECK:   return 0;
; CHECK: }
