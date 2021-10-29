; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -xlntbgen -xlntop="foo" -S -xlntbdummynames="%t" && cat %t | FileCheck %s 

define void @foo([32 x float]* %A) {
  ret void
}

; CHECK: void foo(float A[32]) {
; CHECK: }
; CHECK: int main() {
; CHECK:   static float A[32];
; CHECK:   foo(A);
; CHECK:   return 0;
; CHECK: }
