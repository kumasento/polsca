; RUN: opt < %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -xlnmath -S | FileCheck %s 

; CHECK: double @fnegOnDouble
define double @fnegOnDouble(double %0) {
bb:
  ; CHECK: %[[VAL:.*]] = fsub double -0{{.*}}, %0
  %1 = fneg double %0
  ; CHECK: ret double %[[VAL]]
  ret double %1
}

; CHECK: float @fnegOnFloat
define float @fnegOnFloat(float %0) {
bb:
  ; CHECK: %[[VAL:.*]] = fsub float -0{{.*}}, %0
  %1 = fneg float %0
  ; CHECK: ret float %[[VAL]]
  ret float %1
}
