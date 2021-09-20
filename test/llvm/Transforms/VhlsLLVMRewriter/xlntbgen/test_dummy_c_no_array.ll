; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -xlntbgen -xlntop="foo" -S -xlntbdummynames="%t" && cat %t | FileCheck %s 

define void @foo(i32 %n) {
  ret void
}

; CHECK: void foo(int n) {
; CHECK: }
; CHECK: int main() {
; CHECK:   int n;
; CHECK:   foo(n);
; CHECK:   return 0;
; CHECK: }
