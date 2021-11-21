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

#define DEBUG_TYPE "simp-part"

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
  llvm::DenseMap<AffineExpr, AffineExpr> replacement;

  Value dim = nullptr;
  result.walk([&](AffineExpr e) {
    if (dim) // Found one is fine.
      return;

    auto expr = e.dyn_cast<AffineBinaryOpExpr>();
    if (!expr)
      return;
    if (expr.getKind() != AffineExprKind::FloorDiv)
      return;

    LLVM_DEBUG(dbgs() << "Expr to simplify: " << expr << '\n');
    AffineExpr offset, factor, dimOrSymbol;
    if (!matchOffset(expr, offset, factor, dimOrSymbol))
      return;

    Value dstIndex = getOperandByAffineExpr(avm, dimOrSymbol);
    if (!dstIndex)
      return;
    if (!isForInductionVar(dstIndex))
      return;

    auto forOp = getForInductionVarOwner(dstIndex);
    auto lbMap = filterExtraConstantResults(forOp.getLowerBoundMap());
    auto ubMap = filterExtraConstantResults(forOp.getUpperBoundMap());

    if (lbMap.getNumResults() != ubMap.getNumResults() ||
        lbMap.getNumResults() != 1)
      return;
    if (lbMap.getNumDims() != ubMap.getNumDims() ||
        lbMap.getNumSymbols() != ubMap.getNumSymbols() ||
        lbMap.getNumSymbols() + lbMap.getNumDims() != 1)
      return;

    LLVM_DEBUG(dbgs() << "To replace: " << dstIndex << "\n");

    AffineExpr newLbExpr =
        simplifyAffineExpr(lbMap.getResult(0).floorDiv(factor),
                           lbMap.getNumDims(), lbMap.getNumSymbols());
    AffineExpr newUbExpr =
        simplifyAffineExpr((ubMap.getResult(0) - 1).floorDiv(factor),
                           ubMap.getNumDims(), ubMap.getNumSymbols());
    LLVM_DEBUG(dbgs() << "New LB: " << newLbExpr << '\n');
    LLVM_DEBUG(dbgs() << "New UB: " << newUbExpr << '\n');

    Value lbSrcIndex =
        getValue(lbMap, forOp.getLowerBoundOperands(), newLbExpr);
    Value ubSrcIndex =
        getValue(ubMap, forOp.getUpperBoundOperands(), newUbExpr);
    if (!lbSrcIndex || lbSrcIndex != ubSrcIndex)
      return;

    AffineExpr newExpr = getAffineDimExpr(dims.size(), ctx);
    replacement[e] = newExpr;
    dim = lbSrcIndex;
  });

  if (!dim)
    return nullptr;

  dims.push_back(dim);
  return result.replace(replacement);
}

static std::pair<AffineExpr, AffineExpr>
splitOutTermWithFactor(AffineExpr expr, int64_t factor) {
  /// TODO: work with multiple terms of the same factor.
  if (expr.isMultipleOf(factor))
    return {nullptr, nullptr};
  AffineExpr term = nullptr;
  expr.walk([&](AffineExpr e) {
    if (e.isMultipleOf(factor))
      term = e;
  });

  if (!term)
    return {nullptr, nullptr};

  llvm::DenseMap<AffineExpr, AffineExpr> replacement;
  replacement[term] = getAffineConstantExpr(0, expr.getContext());
  AffineExpr rem = expr.replace(replacement);

  return {term, rem};
}

/// Match and replace x floordiv B if x is within 0 to B
static AffineExpr
simplifyFloorDivIfWithinBound(AffineExpr result, const AffineValueMap &avm,
                              ArrayRef<mlir::AffineForOp> loops,
                              MLIRContext *ctx) {
  llvm::DenseMap<AffineExpr, AffineExpr> replacement;
  llvm::DenseMap<AffineExpr, AffineExpr> lbCstRep, ubCstRep;
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

    auto dim = getAffineDimExpr(p.index(), ctx);
    lbCstRep[dim] = getAffineConstantExpr(lb, ctx);
    ubCstRep[dim] = getAffineConstantExpr(ub, ctx) - 1;

    if (p.index() >= avm.getNumDims())
      continue;
    if (lb < 0 || lb >= ub)
      continue;
    AffineExpr pattern = getAffineDimExpr(p.index(), ctx)
                             .floorDiv(getAffineConstantExpr(ub, ctx));
    LLVM_DEBUG(dbgs() << " -> Patter to replace: " << pattern << "\n");
    replacement.insert({pattern, getAffineConstantExpr(0, ctx)});
  }

  // Try to replace the combination of IVs as a whole.
  result.walk([&](AffineExpr expr) {
    if (auto e = expr.dyn_cast<AffineBinaryOpExpr>())
      if (e.getKind() == AffineExprKind::FloorDiv) {
        auto lhs = e.getLHS().dyn_cast<AffineBinaryOpExpr>();
        if (!lhs)
          return;
        auto lhsLb = lhs.replace(lbCstRep).dyn_cast<AffineConstantExpr>();
        auto lhsUb = lhs.replace(ubCstRep).dyn_cast<AffineConstantExpr>();
        if (!lhsLb || !lhsUb)
          return;

        if (lhsLb.getValue() < 0 ||
            lhsUb.getValue() >=
                e.getRHS().dyn_cast<AffineConstantExpr>().getValue())
          return;

        replacement.insert({expr, getAffineConstantExpr(0, ctx)});
      }
  });

  // Another case
  result.walk([&](AffineExpr expr) {
    if (auto e = expr.dyn_cast<AffineBinaryOpExpr>())
      if (e.getKind() == AffineExprKind::FloorDiv) {
        auto p = splitOutTermWithFactor(
            e.getLHS(), e.getRHS().cast<AffineConstantExpr>().getValue());
        AffineExpr term, rem;
        std::tie(term, rem) = p;

        if (!term || !rem)
          return;

        // Handle rem;
        auto lhsLb = rem.replace(lbCstRep).dyn_cast<AffineConstantExpr>();
        auto lhsUb = rem.replace(ubCstRep).dyn_cast<AffineConstantExpr>();
        if (!lhsLb || !lhsUb)
          return;

        if (lhsLb.getValue() < 0 ||
            lhsUb.getValue() >=
                e.getRHS().dyn_cast<AffineConstantExpr>().getValue())
          return;

        replacement.insert({expr, term.floorDiv(e.getRHS())});
      }
  });

  return result.replace(replacement);
}

/// Match and replace x mod B -> x if x is within [a, B) and a >= 0
static AffineExpr simplifyModIfWithinBound(AffineExpr result,
                                           const AffineValueMap &avm,
                                           ArrayRef<mlir::AffineForOp> loops,
                                           MLIRContext *ctx) {

  llvm::DenseMap<AffineExpr, AffineExpr> replacement;
  llvm::DenseMap<AffineExpr, AffineExpr> lbCstRep, ubCstRep;
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

    auto dim = getAffineDimExpr(p.index(), ctx);
    lbCstRep[dim] = getAffineConstantExpr(lb, ctx);
    ubCstRep[dim] = getAffineConstantExpr(ub, ctx) - 1;

    if (p.index() >= avm.getNumDims())
      continue;
    if (lb < 0 || lb >= ub)
      continue;

    dim = getAffineDimExpr(p.index(), ctx);
    AffineExpr pattern = dim % getAffineConstantExpr(ub, ctx);
    LLVM_DEBUG(dbgs() << " -> Patter to replace: " << pattern << "\n");
    replacement.insert({pattern, dim});
  }

  // Try to replace the combination of IVs as a whole.
  result.walk([&](AffineExpr expr) {
    if (auto e = expr.dyn_cast<AffineBinaryOpExpr>())
      if (e.getKind() == AffineExprKind::Mod) {
        auto lhs = e.getLHS().dyn_cast<AffineBinaryOpExpr>();
        if (!lhs)
          return;
        auto lhsLb = lhs.replace(lbCstRep).dyn_cast<AffineConstantExpr>();
        auto lhsUb = lhs.replace(ubCstRep).dyn_cast<AffineConstantExpr>();
        if (!lhsLb || !lhsUb)
          return;

        if (lhsLb.getValue() < 0 ||
            lhsUb.getValue() >=
                e.getRHS().dyn_cast<AffineConstantExpr>().getValue())
          return;

        replacement.insert({expr, e.getLHS()});
      }
  });

  return result.replace(replacement);
}

static LogicalResult simplifyPartitionAccess(Operation *op) {
  MLIRContext *ctx = op->getContext();

  MemRefAccess access(op);
  AffineValueMap avm;
  access.getAccessMap(&avm);

  SmallVector<mlir::AffineForOp> forOps;
  getLoopIVs(*op, &forOps);

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
    results[i] = simplifyFloorDivIfWithinBound(results[i], avm, forOps, ctx);

    // Case #3 - x mod B if x is within [0, B)
    results[i] = simplifyModIfWithinBound(results[i], avm, forOps, ctx);
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
