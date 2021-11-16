//===- Utils.h - Utility functions ------------------ C++-===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace phism {

/// Get the top function for the hardware design.
::mlir::FuncOp getTopFunction(::mlir::ModuleOp m);
::mlir::Value expandAffineExpr(::mlir::OpBuilder &builder, ::mlir::Location loc,
                               ::mlir::AffineExpr expr,
                               ::mlir::ValueRange dimValues,
                               ::mlir::ValueRange symbolValues);
::mlir::AffineMap filterExtraConstantResults(::mlir::AffineMap affMap);
::mlir::FuncOp findPhismTop(::mlir::ModuleOp m);
void getFunctionsToKeep(::mlir::ModuleOp m, ::mlir::FuncOp top,
                        ::llvm::SmallPtrSetImpl<::mlir::FuncOp> &keep);

std::pair<::mlir::Operation *, ::mlir::Operation *>
    outlineFunction(::llvm::MutableArrayRef<::mlir::Operation *>,
                    ::llvm::StringRef, ::mlir::ModuleOp);

// void getArgs(::llvm::ArrayRef<::mlir::Operation *>,
//              ::llvm::SetVector<::mlir::Value> &);
// std::pair<::mlir::FuncOp, ::mlir::BlockAndValueMapping>
// createCallee(::llvm::MutableArrayRef<::mlir::Operation *>, int,
// ::mlir::FuncOp,
//              ::mlir::OpBuilder &);
// ::mlir::CallOp createCaller(::llvm::MutableArrayRef<::mlir::Operation *>,
//                             ::mlir::FuncOp, ::mlir::BlockAndValueMapping,
//                             ::mlir::OpBuilder &);
} // namespace phism
