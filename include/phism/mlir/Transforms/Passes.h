#ifndef PHISM_MLIR_TRANSFORMS_PASSES_H
#define PHISM_MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace phism {

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createLoopBoundHoistingPass();

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createEliminateAffineLoadStorePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "phism/mlir/Transforms/Passes.h.inc"

} // namespace phism

#endif
