/// LoadSwitch.cc

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
#include "mlir/IR/IntegerSet.h"
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

#define DEBUG_TYPE "load-switch"

using namespace llvm;
using namespace mlir;
using namespace phism;
static void loadSwitchFloorDiv(mlir::AffineLoadOp load) {
  AffineMap am = load.getAffineMap();

  SmallVector<int64_t> offsets, factors;
  SmallVector<AffineExpr> dimOrSymbolExprs;
  SmallVector<Value> dimOrSymbols;
  for (AffineExpr result : am.getResults()) {
    AffineExpr offset, factor, dimOrSymbol;
    if (!matchOffset(result, offset, factor, dimOrSymbol))
      break;
    if (!offset)
      break;

    LLVM_DEBUG(dbgs() << "Matched " << result << '\n');
    offsets.push_back(offset.cast<AffineConstantExpr>().getValue());
    factors.push_back(factor.cast<AffineConstantExpr>().getValue());
    dimOrSymbolExprs.push_back(dimOrSymbol);
    dimOrSymbols.push_back(getOperandByAffineExpr(load, dimOrSymbol));
  }

  if (offsets.empty())
    return;

  LLVM_DEBUG({
    dbgs() << "All offsets: {";
    interleaveComma(offsets, dbgs());
    dbgs() << "}\n";
  });

  OpBuilder b(load.getContext());
  b.setInsertionPoint(load);

  std::function<Operation *(unsigned, SmallVector<bool>)> process =
      [&](unsigned i, SmallVector<bool> conds) -> Operation * {
    if (i == factors.size()) {
      // Should create the load operation here.
      SmallVector<AffineExpr> results{am.getResults().begin(),
                                      am.getResults().end()};
      for (unsigned j = 0; j < factors.size(); ++j)
        results[j] =
            conds[j] ? (dimOrSymbolExprs[j].floorDiv(factors[j]) + offsets[j])
                     : dimOrSymbolExprs[j].floorDiv(factors[j]);

      AffineMap newAm = AffineMap::get(am.getNumDims(), am.getNumSymbols(),
                                       results, b.getContext());
      return b.create<mlir::AffineLoadOp>(load.getLoc(), newAm,
                                          load.getOperands());
    } else {
      // Create conditions.
      bool isDim = dimOrSymbolExprs[i].isa<mlir::AffineDimExpr>();
      AffineExpr modEq =
          (isDim ? b.getAffineDimExpr(0) : b.getAffineSymbolExpr(0)) %
          b.getAffineConstantExpr(factors[i]);
      IntegerSet iset = IntegerSet::get(/*numDims=*/isDim,
                                        /*numSymbols=*/!isDim, {modEq}, {true});
      auto ifOp = b.create<mlir::AffineIfOp>(
          load.getLoc(), /*resultTypes=*/TypeRange(load.getResult().getType()),
          /*set=*/iset, /*args=*/ValueRange(dimOrSymbols[i]),
          /*withElseRegion=*/true);

      // Then block
      b.setInsertionPointToStart(ifOp.getThenBlock());
      conds.push_back(true);
      Operation *result = process(i + 1, conds);
      b.create<mlir::AffineYieldOp>(load.getLoc(), result->getResult(0));

      // Else block
      b.setInsertionPointToStart(ifOp.getElseBlock());
      conds.pop_back();
      conds.push_back(false);
      result = process(i + 1, conds);
      b.create<mlir::AffineYieldOp>(load.getLoc(), result->getResult(0));

      return ifOp;
    }
  };

  auto newLoadResult = process(0, {});
  LLVM_DEBUG(dbgs() << "Created new load op: \n" << (*newLoadResult) << '\n');

  load.replaceAllUsesWith(newLoadResult);
  load.erase();
}

namespace {
/// TODO: extend to store?
struct LoadSwitchPass : public ::phism::LoadSwitchBase<LoadSwitchPass> {
  void runOnFunction() override {
    FuncOp f = getFunction();
    OpBuilder b(f.getContext());

    SmallVector<mlir::AffineLoadOp> loads;
    f.walk([&](mlir::AffineLoadOp loadOp) { loads.push_back(loadOp); });

    for (mlir::AffineLoadOp load : loads) {
      loadSwitchFloorDiv(load);
    }
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
phism::createLoadSwitchPass() {
  return std::make_unique<LoadSwitchPass>();
}
