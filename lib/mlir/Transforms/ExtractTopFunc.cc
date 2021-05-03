//===- ExtractTopFunc.cc ----------------------------------------*- C++ -*-===//
//
// This file implements a pass that extract the specified top function out of
// the given module.
//
//===----------------------------------------------------------------------===//

#include "phism/mlir/Transforms/PhismTransforms.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace llvm;
using namespace phism;

/// Union the given function and all the others it calls into 'keep'.
static void unionCallee(FuncOp f, ModuleOp m,
                        SmallPtrSetImpl<Operation *> &keep) {
  std::deque<Operation *> worklist;
  worklist.push_back(f);
  keep.insert(f);

  // A breadth first search on the call graph, starting from f.
  while (!worklist.empty()) {
    Operation *g = worklist.front();
    worklist.pop_front();

    g->walk([&](Operation *op) {
      if (CallOp callOp = dyn_cast<CallOp>(op)) {
        FuncOp callee =
            dyn_cast_or_null<FuncOp>(m.lookupSymbol(callOp.getCallee()));
        assert(callee && "A callee cannot be resolved in the module.");

        if (!keep.contains(callee)) {
          worklist.push_back(callee);
          keep.insert(callee);
        }
      }
    });
  }
}

namespace {

/// Extract the specified top function out of its module.
struct ExtractTopFuncPass
    : public PassWrapper<ExtractTopFuncPass, OperationPass<ModuleOp>> {

  ExtractTopFuncPass() = default;
  ExtractTopFuncPass(const ExtractTopFuncPass &pass) {}

  Option<std::string> topFuncName{
      *this, "name",
      llvm::cl::desc("Name of the top function to be extracted.")};

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    assert(!topFuncName.empty() && "-name should be specified.");

    FuncOp f = dyn_cast_or_null<FuncOp>(m.lookupSymbol(topFuncName));
    assert(f && "Given name cannot be found in the module as a FuncOp.");

    SmallPtrSet<Operation *, 4> keep;
    unionCallee(f, m, keep);

    m.walk([&](FuncOp g) {
      if (!keep.contains(g))
        g.erase();
      else // Enforce every extracted function to be public
        g.setVisibility(SymbolTable::Visibility::Public);
    });
  }
};

} // namespace

void phism::registerExtractTopFuncPass() {
  PassRegistration<ExtractTopFuncPass>(
      "extract-top-func", "Extract the top function out of its module.");
}
