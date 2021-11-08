//===- SimplifyPartitionAccess.cc - simplify access ------------------ C++-===//

#include "PassDetail.h"
#include "phism/mlir/Transforms/PhismTransforms.h"
#include "phism/mlir/Transforms/Utils.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"

#include <fstream>
#include <queue>
#include <set>

#define DEBUG_TYPE "array-partition"

using namespace mlir;
using namespace llvm;
using namespace phism;

static Value getValue(const AffineMap &avm, OperandRange operands,
                      AffineExpr expr) {
  if (auto dim = expr.dyn_cast<AffineDimExpr>())
    return operands[dim.getPosition()];
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>())
    return operands[sym.getPosition() + avm.getNumDims()];

  return nullptr;
}

static LogicalResult simplifyFloorDivAccess(Operation *op) {
  MLIRContext *ctx = op->getContext();

  MemRefAccess access(op);
  AffineValueMap avm;
  access.getAccessMap(&avm);

  const AffineMap &am = avm.getAffineMap();

  SmallVector<AffineExpr> results;
  for (unsigned i = 0; i < avm.getNumResults(); ++i)
    results.push_back(avm.getResult(i));

  SmallVector<Value> dims, symbols;
  for (unsigned i = 0; i < avm.getNumDims(); ++i)
    dims.push_back(avm.getOperand(i));
  for (unsigned i = 0; i < avm.getNumSymbols(); ++i)
    symbols.push_back(avm.getOperand(i + avm.getNumDims()));

  for (unsigned i = 0; i < am.getNumResults(); ++i) {
    AffineExpr result = am.getResult(i);
    if (auto expr = result.dyn_cast<AffineBinaryOpExpr>()) {
      if (expr.getKind() != AffineExprKind::FloorDiv)
        continue;

      AffineConstantExpr denom = expr.getRHS().dyn_cast<AffineConstantExpr>();
      if (!denom)
        continue;

      Value dstIndex;
      if (auto dim = expr.getLHS().dyn_cast<AffineDimExpr>())
        dstIndex = avm.getOperand(dim.getPosition());
      else if (auto symbol = expr.getLHS().dyn_cast<AffineSymbolExpr>())
        dstIndex = avm.getOperand(symbol.getPosition() + am.getNumDims());

      if (!dstIndex)
        continue;
      if (!isForInductionVar(dstIndex))
        continue;

      auto forOp = getForInductionVarOwner(dstIndex);
      auto lbMap = filterExtraConstantResults(forOp.getLowerBoundMap());
      auto ubMap = filterExtraConstantResults(forOp.getUpperBoundMap());

      if (lbMap.getNumResults() != ubMap.getNumResults() ||
          lbMap.getNumResults() != 1)
        continue;
      if (lbMap.getNumDims() != ubMap.getNumDims() ||
          lbMap.getNumSymbols() != ubMap.getNumSymbols() ||
          lbMap.getNumSymbols() + lbMap.getNumDims() != 1)
        continue;

      LLVM_DEBUG(dbgs() << "To replace: " << dstIndex << "\n");

      AffineExpr newLbExpr =
          simplifyAffineExpr(lbMap.getResult(0).floorDiv(denom),
                             lbMap.getNumDims(), lbMap.getNumSymbols());
      AffineExpr newUbExpr =
          simplifyAffineExpr((ubMap.getResult(0) - 1).floorDiv(denom),
                             ubMap.getNumDims(), ubMap.getNumSymbols());
      LLVM_DEBUG(dbgs() << "New LB: " << newLbExpr << '\n');
      LLVM_DEBUG(dbgs() << "New UB: " << newUbExpr << '\n');

      Value lbSrcIndex =
          getValue(lbMap, forOp.getLowerBoundOperands(), newLbExpr);
      Value ubSrcIndex =
          getValue(ubMap, forOp.getUpperBoundOperands(), newUbExpr);
      if (!lbSrcIndex || lbSrcIndex != ubSrcIndex)
        continue;

      results[i] = getAffineDimExpr(dims.size(), ctx);
      dims.push_back(lbSrcIndex);
    }
  }

  SmallVector<Value> operands{op->getOperand(0)};
  if (isa<mlir::AffineWriteOpInterface>(op))
    operands.push_back(op->getOperand(1));
  operands.append(dims);
  operands.append(symbols);

  op->setOperands(operands);

  AffineMap newAffMap =
      AffineMap::get(dims.size(), symbols.size(), results, ctx);
  if (auto loadOp = dyn_cast<mlir::AffineLoadOp>(op))
    loadOp->setAttr(loadOp.getMapAttrName(), AffineMapAttr::get(newAffMap));
  if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(op))
    storeOp->setAttr(storeOp.getMapAttrName(), AffineMapAttr::get(newAffMap));

  return success();
}

namespace {
struct SimplifyPartitionAccessPass
    : public phism::SimplifyPartitionAccessBase<SimplifyPartitionAccessPass> {
  void runOnFunction() override {
    FuncOp f = getFunction();
    OpBuilder b(f.getContext());

    f.walk([&](Operation *op) {
      // TODO: memref load/store
      if (!isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op))
        return;
      if (failed(simplifyFloorDivAccess(op)))
        return signalPassFailure();
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
phism::createSimplifyPartitionAccessPass() {
  return std::make_unique<SimplifyPartitionAccessPass>();
}
