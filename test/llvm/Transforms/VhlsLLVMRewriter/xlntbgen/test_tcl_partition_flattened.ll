; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -xlntbgen -xlntop="foo" -S -xlntbtclnames="%t" -xlntbdummynames="%t.dummy.c" -xln-ap-flattened -xln-ap-enabled && cat %t | FileCheck %s 

define void @foo(i32 %n, [32 x float]* %A, [4 x [3 x [6 x [8 x i32]]]]* %B) {
  ret void
}

; CHECK: open_project -reset tb
; CHECK: add_files 
; CHECK: add_files -tb 
; CHECK: set_top foo
; CHECK: open_solution -reset solution1
; CHECK: set_part 
; CHECK: create_clock -period "100MHz"
; CHECK: set_directive_array_partition -dim 1 -factor 4 -type block "foo" B
; CHECK: set_directive_interface foo B -mode ap_memory -storage_type ram_2p
