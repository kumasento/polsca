; RUN: opt %s -enable-new-pm=0 -load ${PHISM_LIBS_DIR}/VhlsLLVMRewriter.so -mem2arr -S | FileCheck %s 

; ModuleID = '<stdin>'
source_filename = "LLVMDialectModule"
target triple = "x86_64-unknown-linux-gnu"

declare i8* @malloc(i64)

declare void @free(i8*)

define i32 @main() {
  %1 = call i8* @malloc(i64 ptrtoint (double* getelementptr (double, double* null, i64 200) to i64))
  %2 = bitcast i8* %1 to double*
  %3 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } undef, double* %2, 0
  %4 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %3, double* %2, 1
  %5 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %4, i64 0, 2
  %6 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %5, i64 10, 3, 0
  %7 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %6, i64 20, 3, 1
  %8 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %7, i64 20, 4, 0
  %9 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %8, i64 1, 4, 1
  %10 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1
  call void @update(double* %10)
  ret i32 0
}

define void @update(double* %0) {
  %2 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } undef, double* %0, 0
  %3 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %2, double* %0, 1
  %4 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %4, i64 10, 3, 0
  %6 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %5, i64 20, 4, 0
  %7 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %6, i64 20, 3, 1
  %8 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  br label %9

9:                                                ; preds = %28, %1
  %10 = phi i64 [ %29, %28 ], [ 0, %1 ]
  %11 = icmp slt i64 %10, 10
  br i1 %11, label %12, label %30

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %16, %12
  %14 = phi i64 [ %27, %16 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 20
  br i1 %15, label %16, label %28

16:                                               ; preds = %13
  %17 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %8, 1
  %18 = mul i64 %10, 20
  %19 = add i64 %18, %14
  %20 = getelementptr double, double* %17, i64 %19
  %21 = load double, double* %20, align 8
  %22 = fmul double %21, 1.000000e+02
  %23 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %8, 1
  %24 = mul i64 %10, 20
  %25 = add i64 %24, %14
  %26 = getelementptr double, double* %23, i64 %25
  store double %22, double* %26, align 8
  %27 = add i64 %14, 1
  br label %13

28:                                               ; preds = %13
  %29 = add i64 %10, 1
  br label %9

30:                                               ; preds = %9
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: define i32 @main()
; CHECK-NEXT: %[[v1:.*]] = call i8* @malloc
; CHECK-NEXT: %[[v2:.*]] = bitcast i8* %[[v1]] to [10 x [20 x double]]*
; CHECK-NEXT: call void @update([10 x [20 x double]]* %2)
