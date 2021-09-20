; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -xlntbgen -xlntop="foo" -S -xlntbdummynames="%t" && cat %t | FileCheck %s 

define void @foo(i32 %n, [32 x [24 x float]]* %A) {
  ret void
}

; CHECK: void foo(int n, float A[32][24]) {
; CHECK:   A[n + 1][n + 1] += A[n][n];
; CHECK: }
; CHECK: int main() {
; CHECK:   int n;
; CHECK:   float A[32][24];
; CHECK:   foo(n, A);
; CHECK:   return 0;
; CHECK: }
