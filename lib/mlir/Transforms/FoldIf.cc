//===- FoldIf.cc - Fold affine.if into select ---------------- C++-===//
#include "phism/mlir/Transforms/PhismTransforms.h"
#include "phism/mlir/Transforms/Utils.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

#include <fstream>
#include <queue>
#include <set>

#define DEBUG_TYPE "fold-if"

using namespace mlir;
using namespace llvm;
using namespace phism;

static void addEligibleAffineIfOps(FuncOp f,
                                   SmallVector<mlir::AffineIfOp> &ifOps) {
  f.walk([&](mlir::AffineIfOp ifOp) {
    LLVM_DEBUG(dbgs() << "Num regions: " << ifOp.getNumRegions()
                      << ", num blocks: "
                      << ifOp.getBodyRegion().getBlocks().size() << '\n');
    // There should be a single region with a single block.
    if (ifOp.hasElse() || ifOp.getBodyRegion().getBlocks().size() > 1)
      return;
    // TODO: other conditions
    ifOps.push_back(ifOp);
  });
}

static Value createAffineIfCond(mlir::AffineIfOp ifOp, OpBuilder &b) {
  Location loc = ifOp.getLoc();

  IntegerSet integerSet = ifOp.getIntegerSet();
  Value zeroConstant = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 8> operands(ifOp.getOperands());
  auto operandsRef = llvm::makeArrayRef(operands);

  Value cond = nullptr;

  for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
    AffineExpr constraintExpr = integerSet.getConstraint(i);
    bool isEquality = integerSet.isEq(i);

    auto numDims = integerSet.getNumDims();
    Value affResult = expandAffineExpr(b, loc, constraintExpr,
                                       operandsRef.take_front(numDims),
                                       operandsRef.drop_front(numDims));
    if (!affResult)
      return nullptr;

    auto pred =
        isEquality ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
    Value cmpVal =
        b.create<mlir::arith::CmpIOp>(loc, pred, affResult, zeroConstant);
    cond = cond ? b.create<mlir::arith::AndIOp>(loc, cond, cmpVal).getResult()
                : cmpVal;
  }

  return cond ? cond
              : b.create<arith::ConstantIntOp>(loc, /*value=*/1, /*width=*/1);
}

/// The value to be store is either the original value to be stored, or the
/// current value at this given address.
static LogicalResult process(mlir::AffineStoreOp storeOp, Value cond,
                             BlockAndValueMapping &vmap, OpBuilder &b) {
  Location loc = storeOp.getLoc();

  Value memref = storeOp.getMemRef();
  AffineMap affMap = storeOp.getAffineMap();
  SmallVector<Value, 8> mapOperands(storeOp.getMapOperands());

  Value orig = b.create<mlir::AffineLoadOp>(loc, memref, affMap, mapOperands);
  Value toStore = b.create<SelectOp>(
      loc, cond, vmap.lookup(storeOp.getValueToStore()), orig);

  b.create<mlir::AffineStoreOp>(loc, toStore, memref, affMap, mapOperands);

  return success();
}

/// TODO: filter invalid operations.
/// TODO: affine.load might load from invalid address.
static LogicalResult process(mlir::AffineIfOp ifOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(ifOp);

  // Turn the if-condition evaluation result to a single value.
  // This implementation is inspired by AffineIfLowering from AffineToStandard.
  Value cond = createAffineIfCond(ifOp, b);
  if (!cond)
    return failure();

  BlockAndValueMapping vmap;
  for (Operation &op : ifOp.getBody()->getOperations()) {
    if (isa<mlir::AffineYieldOp>(op))
      continue;

    if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(op)) {
      if (failed(process(storeOp, cond, vmap, b)))
        return failure();
    } else {
      b.clone(op, vmap);
    }
  }

  ifOp.erase();

  return success();
}

namespace {
struct FoldIfPass : PassWrapper<FoldIfPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<mlir::AffineIfOp> ifOps;
    m.walk([&](FuncOp f) { addEligibleAffineIfOps(f, ifOps); });

    for (mlir::AffineIfOp ifOp : ifOps)
      if (failed(process(ifOp, b)))
        LLVM_DEBUG(dbgs() << "Failed to process: " << ifOp << '\n');
  }
};
} // namespace

namespace phism {
void registerFoldIfPasses() {

  PassPipelineRegistration<>("fold-if", "Fold affine.if",
                             [&](OpPassManager &pm) {
                               pm.addPass(std::make_unique<FoldIfPass>());
                               //  pm.addPass(createCanonicalizerPass());
                             });
}
} // namespace phism
