// RUN: phism-opt %s -convert-std-to-llvm -mem2ptr-call | FileCheck %s

// CHECK-LABEL: func @multi_memrefs_and_results
func @multi_memrefs_and_results(%A: memref<1xf32>, %B: memref<1xf32>) -> f32 {
  %c0 = constant 0 : index

  %0 = load %A[%c0] : memref<1xf32>
  %1 = addf %0, %0 : f32
  store %1, %B[%c0] : memref<1xf32>

  return %1 : f32

  // CHECK: %[[PTR0:.*]] = llvm.extractvalue 
  // CHECK-NEXT: %[[PTR1:.*]] = llvm.extractvalue 
  // CHECK-NEXT: %[[RET:.*]] = llvm.call @[[CALLEE:.*]](%[[PTR0]], %[[PTR1]])
  // CHECK-NEXT: llvm.return %[[RET]]
}

// CHECK: func @[[CALLEE]]
// CHECK-NEXT: %[[VAL0:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[VAL1:.*]] = llvm.getelementptr
// CHECK-NEXT: %[[VAL2:.*]] = llvm.load %[[VAL1]]
// CHECK-NEXT: %[[VAL3:.*]] = llvm.fadd %[[VAL2]], %[[VAL2]]
// CHECK-NEXT: %[[VAL4:.*]] = llvm.getelementptr 
// CHECK-NEXT: llvm.store %[[VAL3]], %[[VAL4]]
