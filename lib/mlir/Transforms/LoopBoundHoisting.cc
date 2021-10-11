//===- ExtractTopFunc.cc ----------------------------------------*- C++ -*-===//
//
// This file implements a pass that extract the specified top function out of
// the given module.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace llvm;
using namespace phism;

#define PASS_NAME "loop-bound-hoisting"
#define DEBUG_TYPE PASS_NAME

static bool isHoistable(scf::ForOp forOp) {
  LLVM_DEBUG(dbgs() << "Examining whetehr " << forOp << " is hoistable.\n");

  SmallVector<Operation *> ops;
  for (auto &op : forOp.getBody()->getOperations())
    ops.push_back(&op);
  ops.pop_back(); // scf.yield

  // There should be at least two operations (including the inner scf.for).
  if (ops.size() <= 1)
    return false;
  // The last operation within the body should be an scf.for.
  if (!isa<scf::ForOp>(ops.back()))
    return false;
  // The other operations cannot be scf.for.
  if (any_of(make_range(ops.begin(), std::prev(ops.end())),
             [](Operation *op) { return isa<scf::ForOp>(op); }))
    return false;

  // The operations in-between either define the inner loop's bounds, or only
  // used to calculate those bounds.
  scf::ForOp innerLoop = cast<scf::ForOp>(ops.back());
  for (Operation *op : ops) {
    if (op == innerLoop)
      continue;

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        // Serve as the bounds.
        if (user == innerLoop)
          continue;
        // The user should be at the same level of the innerLoop.
        if (user->getParentOp() != forOp)
          return false;
      }
    }
  }

  return true;
}

/// A scf::ForOp is hoistable if it has another scf.for within it, which is the
/// only scf.for in the body, and has loop bound operations in between
/// (prologue).
static scf::ForOp findHoistableForOp(FuncOp f) {
  scf::ForOp candidate = nullptr;

  f.walk([&](scf::ForOp forOp) {
    if (!candidate && isHoistable(forOp)) {
      candidate = forOp;
      return;
    }
  });
  return candidate;
}

static void hoistInnerLoop(scf::ForOp forOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(forOp);

  SmallVector<Operation *> toErase;

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isa<scf::ForOp, scf::YieldOp>(op))
      continue;

    Operation *cloned = b.clone(op);
    op.replaceAllUsesWith(cloned);

    toErase.push_back(&op);
  }

  for (Operation *op : toErase)
    op->erase();
}

namespace {
struct LoopBoundHoistingPass
    : public LoopBoundHoistingBase<LoopBoundHoistingPass> {
  void runOnFunction() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    // Iterative algorithm.
    scf::ForOp forOp;
    int maxIterations = 100, iter = 0;
    while ((forOp = findHoistableForOp(f))) {
      LLVM_DEBUG(dbgs() << "Hoistable scf.for: " << forOp << '\n');

      hoistInnerLoop(forOp, b);

      if (iter >= maxIterations) {
        LLVM_DEBUG(dbgs() << "Exceeded max iterations " << maxIterations
                          << ", exiting pass ...\n");
        break;
      }

      ++iter;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> phism::createLoopBoundHoistingPass() {
  return std::make_unique<LoopBoundHoistingPass>();
}
