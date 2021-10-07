//===- Utils.h - Utility functions ------------------ C++-===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"

namespace phism {

/// Get the top function for the hardware design.
mlir::FuncOp getTopFunction(mlir::ModuleOp m);
mlir::Value expandAffineExpr(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::AffineExpr expr, mlir::ValueRange dimValues,
                             mlir::ValueRange symbolValues);

} // namespace phism
