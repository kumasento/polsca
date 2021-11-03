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
    Option<"topOnly", "top-only", "bool", /*default=*/"false",
           "Whether should only apply for phism.top">,
    Option<"loadAfterStore", "load-after-store", "bool", /*default=*/"true",
           "Whether to perform load-after-store elimination">
  ];
}

def SplitNonAffine : FunctionPass<"split-non-affine"> {
  let summary = "Split affine and non-affine loops into functions.";
  let constructor = "phism::createSplitNonAffinePass()";
  let options = [
    Option<"topOnly", "top-only", "bool", /*default=*/"false",
           "Whether should only apply for phism.top">,
    Option<"markOnly", "mark-only", "bool", /*default=*/"false",
           "Whether we just annotate those for loops with non affine accesses.">,
    Option<"maxLoopDepth", "max-loop-depth", "int", /*default=*/"0",
           "Below which loop depth should we put affine loops into functions.">
  ];
}

#endif
