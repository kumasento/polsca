; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -xlnram2p -xlntop="foo" -S | FileCheck %s 

define void @foo(i32 %n, [2 x [32 x float]]* %A) {
  ret void
}

; CHECK: define void @foo
; CHECK: [2 x [32 x float]]* "fpga.address.interface"="ap_memory.A" %A
; CHECK: !fpga.adaptor.bram.A !0
; CHECK: !0 = !{!"A", !"ap_memory", i32 666, i32 208, i32 -1}
