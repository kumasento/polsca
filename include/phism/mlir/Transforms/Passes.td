#ifndef PHISM_MLIR_TRANSFORMS_PASSES
#define PHISM_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LoopBoundHoisting : FunctionPass<"loop-bound-hoisting"> {
  let summary = "Hoist loop bound calculation to make perfectly nested loops";
  let constructor = "phism::createLoopBoundHoistingPass()";
}

def EliminateAffineLoadStore : FunctionPass<"eliminate-affine-load-store"> {
  let summary = "Eliminate redundant affine load and store operations.";
  let constructor = "phism::createEliminateAffineLoadStorePass()";
  let options = [
    Option<"loadAfterStore", "load-after-store", "bool", /*default=*/"true",
           "Whether to perform load-after-store elimination">
  ];
}

#endif
