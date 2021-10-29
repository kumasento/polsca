// RUN: mlir-opt -lower-affine -convert-scf-to-std -convert-memref-to-llvm -convert-arith-to-llvm -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -reconcile-unrealized-casts %s | mlir-translate -mlir-to-llvmir | opt -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -strip-debug -mem2arr -S | FileCheck %s 

func @foo(%A: memref<10xf32>) {
  %c0 = arith.constant 0: index
  %0 = memref.load %A[%c0] : memref<10xf32>
  return
}
func @bar(%A: memref<10xf32>) {
  %c0 = arith.constant 0: index
  %0 = memref.load %A[%c0] : memref<10xf32>
  return
}
func @main() {
  %A = memref.alloc() : memref<10xf32>
  call @foo(%A): (memref<10xf32>) -> ()
  call @bar(%A): (memref<10xf32>) -> ()
  return
}

// CHECK-DAG: define void @bar([10 x float]* %{{.*}})

// CHECK-DAG: define void @foo([10 x float]* %{{.*}})

// CHECK: define void @main()
// CHECK-NEXT:   %[[v1:.*]] = call i8* @malloc(i64 ptrtoint (float* getelementptr (float, float* null, i64 10) to i64))
// CHECK-NEXT:   %[[v2:.*]] = bitcast i8* %[[v1]] to [10 x float]*
// CHECK-NEXT:   %[[v3:.*]] = bitcast i8* %[[v1]] to [10 x float]*
// CHECK-NEXT:   call void @foo([10 x float]* %[[v2]])
// CHECK-NEXT:   call void @bar([10 x float]* %[[v3]])
