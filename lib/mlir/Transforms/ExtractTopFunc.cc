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

static constexpr const char *SCOP_CONSTANT_VALUE = "scop.constant_value";

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

/// If any argument to the given function is constant (defined by ConstantOp),
/// we will put its value into the attributes of the function, so that at later
/// stage, once the top function has been extracted out, we can still know what
/// are the bound constant values.
static void annotateConstantArgs(FuncOp f, ModuleOp m, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);

  // Find the caller for f.
  mlir::CallOp caller;
  m.walk([&](mlir::CallOp callOp) {
    if (callOp.getCallee() == f.getName()) {
      assert(!caller &&
             "There should be only one caller for the target function.");
      caller = callOp;
    }
  });

  for (auto arg : enumerate(f.getArguments())) {
    auto val = caller.getOperand(arg.index());
    if (mlir::ConstantOp constantOp =
            dyn_cast<mlir::ConstantOp>(val.getDefiningOp())) {
      f.setArgAttr(arg.index(), SCOP_CONSTANT_VALUE, constantOp.getValue());
    }
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

    annotateConstantArgs(f, m, b);

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

/// --------------------------- ReplaceConstantArguments ----------------------

namespace {

struct ReplaceConstantArgumentsPass
    : public PassWrapper<ReplaceConstantArgumentsPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    m.walk([&](FuncOp funcOp) {
      for (auto arg : enumerate(funcOp.getArguments())) {
        Attribute attr = funcOp.getArgAttr(arg.index(), SCOP_CONSTANT_VALUE);
        if (attr) {
          b.setInsertionPointToStart(&funcOp.getBlocks().front());
          ConstantOp constantOp = b.create<ConstantOp>(funcOp.getLoc(), attr);
          arg.value().replaceAllUsesWith(constantOp);
        }
      }
    });
  }
};
} // namespace

void phism::registerExtractTopFuncPass() {
  PassRegistration<ExtractTopFuncPass>(
      "extract-top-func", "Extract the top function out of its module.");
  PassRegistration<ReplaceConstantArgumentsPass>(
      "replace-constant-arguments",
      "Replace the annotated constant arguments.");
}
