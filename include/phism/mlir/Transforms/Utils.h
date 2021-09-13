//===- Utils.h - Utility functions ------------------ C++-===//

#include "mlir/IR/BuiltinOps.h"

namespace phism {

/// Get the top function for the hardware design.
mlir::FuncOp getTopFunction(mlir::ModuleOp m);

} // namespace phism
