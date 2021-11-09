//===- StripExceptTop.cc ----------------------------------------*- C++ -*-===//
//
// Strip functions except those in the phism.top call graph.
//
//===----------------------------------------------------------------------===//

#include "./PassDetail.h"
#include "phism/mlir/Transforms/PhismTransforms.h"
#include "phism/mlir/Transforms/Utils.h"

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

#include <memory>
#include <queue>

using namespace mlir;
using namespace llvm;
using namespace phism;

namespace {
struct StripExceptTopPass
    : public phism::StripExceptTopBase<StripExceptTopPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    FuncOp top = findPhismTop(m);
    if (!top) {
      errs() << "There is no phism.top found.";
      return signalPassFailure();
    }

    llvm::SetVector<FuncOp> keep;
    std::queue<FuncOp> worklist;

    worklist.push(top);

    while (!worklist.empty()) {
      FuncOp f = worklist.front();
      worklist.pop();

      keep.insert(f);
      f.walk([&](CallOp caller) {
        FuncOp callee = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
        if (callee)
          worklist.push(callee);
      });
    }

    SmallVector<FuncOp> toErase;
    m.walk([&](FuncOp f) {
      if (!keep.count(f))
        toErase.push_back(f);
    });

    for (FuncOp f : toErase)
      f.erase();
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createStripExceptTopPass() {
  return std::make_unique<StripExceptTopPass>();
}
