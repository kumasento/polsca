#ifndef PHISM_MLIR_TRANSFORMS_PASSES
#define PHISM_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LoopBoundHoisting : FunctionPass<"loop-bound-hoisting"> {
  let summary = "Hoist loop bound calculation to make perfectly nested loops";
  let constructor = "phism::createLoopBoundHoistingPass()";
}

#endif
