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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

#include "llvm/ADT/SetVector.h"

#include <deque>

using namespace mlir;
using namespace llvm;
using namespace phism;

static constexpr const char *SCOP_CONSTANT_VALUE = "scop.constant_value";

namespace {
struct PipelineOptions : public mlir::PassPipelineOptions<PipelineOptions> {
  Option<std::string> topFuncName{*this, "name",
                                  llvm::cl::desc("Top function name")};
  Option<bool> keepAll{*this, "keepall",
                       llvm::cl::desc("Keep all the functions.")};
};
} // namespace

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

  std::string topFuncName;
  bool keepAll = false;

  ExtractTopFuncPass() = default;
  ExtractTopFuncPass(const ExtractTopFuncPass &pass) {}
  ExtractTopFuncPass(const PipelineOptions &options)
      : topFuncName{options.topFuncName}, keepAll{options.keepAll} {}

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
      if (!keep.contains(g)) {
        if (!keepAll)
          g.erase();
      } else // Enforce every extracted function to be public
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

/// --------------------------- RemoveEmptyAffineIf ----------------------

namespace {

/// Check if the integer set of affine.if is empty. We don't use the trivial
/// isEmpty test since that would only check the equalities. We try to get an
/// integer sample, and if that doesn't exist, we know the set is empty.
struct RemoveEmptyAffineIfPass
    : public PassWrapper<RemoveEmptyAffineIfPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<AffineIfOp> worklist;

    m.walk([&](AffineIfOp ifOp) {
      FlatAffineConstraints cst(ifOp.getIntegerSet());
      if (!cst.findIntegerSample().hasValue())
        worklist.push_back(ifOp);
      else {
        cst.removeRedundantInequalities();
        IntegerSet ist = cst.getAsIntegerSet(b.getContext());

        // It is not always possible to convert back to a valid (not NULL)
        // IntegerSet from FAC. In that case, we will do nothing.
        if (!ist)
          return;

        ifOp.setIntegerSet(ist);
      }
    });

    for (AffineIfOp ifOp : worklist)
      ifOp.erase();
  }
};
} // namespace

/// --------------------------- PropagateConstants ----------------------

namespace {

/// Replace the arguments of the PEs by constants if they are passed that way.
struct PropagateConstantsPass
    : public PassWrapper<PropagateConstantsPass, OperationPass<ModuleOp>> {

  std::string topName;

  PropagateConstantsPass() = default;
  PropagateConstantsPass(const PropagateConstantsPass &pass) {}
  PropagateConstantsPass(const PipelineOptions &options)
      : topName(options.topFuncName) {}

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // Map from caller to its mapped constant values.
    llvm::DenseMap<Operation *, llvm::DenseMap<unsigned, Value>> constArgs;

    // Look within the top function, go through each caller, and if there is any
    // argument is defined by a constant integer, put that as a record in the
    // dictionary.
    FuncOp top = cast<FuncOp>(m.lookupSymbol(topName));
    top.walk([&](CallOp caller) {
      for (auto arg : enumerate(caller.getArgOperands()))
        if (arg.value().getDefiningOp<arith::ConstantIntOp>()) {
          if (!constArgs[caller].count(arg.index())) {
            constArgs[caller].insert({arg.index(), arg.value()});
          } else {
            // If an argument can be called by different constants, then it
            // cannot be propagated.
            /// TODO: should we better duplicate the PE instances?
            constArgs[caller][arg.index()] = nullptr; // invalid.
          }
        }
    });

    // Replace -
    for (auto &it : constArgs) {
      FuncOp callee =
          cast<FuncOp>(m.lookupSymbol(cast<CallOp>(it.first).getCallee()));
      b.setInsertionPointToStart(&callee.getBlocks().front());
      for (auto &it2 : it.second) {
        unsigned idx;
        Value arg;
        std::tie(idx, arg) = it2;

        if (!arg)
          continue;

        Operation *op = b.clone(*arg.getDefiningOp());
        callee.getArgument(idx).replaceAllUsesWith(op->getResult(0));
      }
    }
  }
};
} // namespace

void phism::registerExtractTopFuncPass() {
  PassPipelineRegistration<PipelineOptions>(
      "extract-top-func", "Extract the top function out of its module.",
      [](OpPassManager &pm, const PipelineOptions &options) {
        pm.addPass(std::make_unique<ExtractTopFuncPass>(options));
      });
  PassPipelineRegistration<PipelineOptions>(
      "replace-constant-arguments", "Replace the annotated constant arguments.",
      [](OpPassManager &pm, const PipelineOptions &options) {
        pm.addPass(std::make_unique<ReplaceConstantArgumentsPass>());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(std::make_unique<RemoveEmptyAffineIfPass>());
        pm.addPass(std::make_unique<PropagateConstantsPass>(options));
        pm.addPass(createCanonicalizerPass());
      });
}
