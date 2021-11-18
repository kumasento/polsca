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

/// Case #1 - (B * x) floordiv B -> B
static AffineExpr
simplifyFloorDivForSameMultiplier(AffineExpr result, const AffineValueMap &avm,
                                  SmallVectorImpl<Value> &dims,
                                  MLIRContext *ctx) {
  auto expr = result.dyn_cast<AffineBinaryOpExpr>();
  if (!expr)
    return nullptr;
  if (expr.getKind() != AffineExprKind::FloorDiv)
    return nullptr;

  AffineConstantExpr denom = expr.getRHS().dyn_cast<AffineConstantExpr>();
  if (!denom)
    return nullptr;

  Value dstIndex;
  if (auto dim = expr.getLHS().dyn_cast<AffineDimExpr>())
    dstIndex = avm.getOperand(dim.getPosition());
  else if (auto symbol = expr.getLHS().dyn_cast<AffineSymbolExpr>())
    dstIndex = avm.getOperand(symbol.getPosition() + avm.getNumDims());

  if (!dstIndex)
    return nullptr;
  if (!isForInductionVar(dstIndex))
    return nullptr;

  auto forOp = getForInductionVarOwner(dstIndex);
  auto lbMap = filterExtraConstantResults(forOp.getLowerBoundMap());
  auto ubMap = filterExtraConstantResults(forOp.getUpperBoundMap());

  if (lbMap.getNumResults() != ubMap.getNumResults() ||
      lbMap.getNumResults() != 1)
    return nullptr;
  if (lbMap.getNumDims() != ubMap.getNumDims() ||
      lbMap.getNumSymbols() != ubMap.getNumSymbols() ||
      lbMap.getNumSymbols() + lbMap.getNumDims() != 1)
    return nullptr;

  LLVM_DEBUG(dbgs() << "To replace: " << dstIndex << "\n");

  AffineExpr newLbExpr =
      simplifyAffineExpr(lbMap.getResult(0).floorDiv(denom), lbMap.getNumDims(),
                         lbMap.getNumSymbols());
  AffineExpr newUbExpr =
      simplifyAffineExpr((ubMap.getResult(0) - 1).floorDiv(denom),
                         ubMap.getNumDims(), ubMap.getNumSymbols());
  LLVM_DEBUG(dbgs() << "New LB: " << newLbExpr << '\n');
  LLVM_DEBUG(dbgs() << "New UB: " << newUbExpr << '\n');

  Value lbSrcIndex = getValue(lbMap, forOp.getLowerBoundOperands(), newLbExpr);
  Value ubSrcIndex = getValue(ubMap, forOp.getUpperBoundOperands(), newUbExpr);
  if (!lbSrcIndex || lbSrcIndex != ubSrcIndex)
    return nullptr;

  AffineExpr newExpr = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(lbSrcIndex);
  return newExpr;
}

/// Match and replace x floordiv B if x is within 0 to B
static AffineExpr simplifyFloorDivIfWithinBound(AffineExpr result,
                                                const AffineValueMap &avm,
                                                MLIRContext *ctx) {
  llvm::DenseMap<AffineExpr, AffineExpr> replacement;
  for (auto p : enumerate(avm.getOperands())) {
    Value operand = p.value();
    if (!isForInductionVar(operand))
      continue;
    auto forOp = getForInductionVarOwner(operand);
    if (!forOp.getLowerBoundMap().isSingleConstant() ||
        !forOp.getLowerBoundMap().isSingleConstant())
      continue;

    auto lb = forOp.getLowerBoundMap().getSingleConstantResult();
    auto ub = forOp.getUpperBoundMap().getSingleConstantResult();

    if (p.index() >= avm.getNumDims())
      continue;
    if (lb < 0)
      continue;
    AffineExpr pattern = getAffineDimExpr(p.index(), ctx)
                             .floorDiv(getAffineConstantExpr(ub, ctx));
    LLVM_DEBUG(dbgs() << " -> Patter to replace: " << pattern << "\n");
    replacement.insert({pattern, getAffineConstantExpr(0, ctx)});
  }

  return result.replace(replacement);
}

static LogicalResult simplifyPartitionAccess(Operation *op) {
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

  // Simplify per result of the input affine map.
  for (unsigned i = 0; i < am.getNumResults(); ++i) {
    // Case #1 - (B * x) floordiv B -> B
    if (AffineExpr expr =
            simplifyFloorDivForSameMultiplier(results[i], avm, dims, ctx))
      results[i] = expr;
    // Case #2 - x floordiv B if x is within [0, B)
    // This is actually a special case of the version above. We distinguish them
    // just for simpler processing.
    if (AffineExpr expr = simplifyFloorDivIfWithinBound(results[i], avm, ctx))
      results[i] = expr;
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
      if (failed(simplifyPartitionAccess(op)))
        return signalPassFailure();
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
phism::createSimplifyPartitionAccessPass() {
  return std::make_unique<SimplifyPartitionAccessPass>();
}
