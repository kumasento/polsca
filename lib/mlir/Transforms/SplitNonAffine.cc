//===- SplitNonAffine.cc ----------------------------------------*- C++ -*-===//
//
// Split affine and non-affine loops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "split-non-affine"

using namespace mlir;
using namespace llvm;
using namespace phism;

/// Find memref.load and memref.store, and mark their parent affine.for to be
/// non-affine.
/// CallOp is also consider non-affine.
static void markNonAffine(FuncOp f, OpBuilder &b) {
  f.walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp, CallOp>(op)) {
      while (Operation *parent = op->getParentOp()) {
        if (isa<FuncOp>(parent))
          break;
        if (auto forOp = dyn_cast<mlir::AffineForOp>(parent))
          forOp->setAttr("scop.non_affine_access", b.getUnitAttr());
        op = parent;
      }
    }
  });
}

static void getArgs(ArrayRef<Operation *> ops, llvm::SetVector<Value> &args) {
  args.clear();

  llvm::SetVector<Operation *> internalOps;
  for (Operation *parentOp : ops) {
    internalOps.insert(parentOp);
    parentOp->walk([&](Operation *op) { internalOps.insert(op); });
    parentOp->walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (!internalOps.contains(defOp))
            args.insert(operand);
        } else if (BlockArgument bArg = operand.dyn_cast<BlockArgument>()) {
          if (!internalOps.contains(bArg.getOwner()->getParentOp()))
            args.insert(operand);
        } else {
          llvm_unreachable("Operand cannot be handled.");
        }
      }
    });
  }
}

static std::pair<FuncOp, BlockAndValueMapping>
createCallee(MutableArrayRef<Operation *> forOps, int id, FuncOp f,
             OpBuilder &b) {
  assert(!forOps.empty());

  ModuleOp m = f->getParentOfType<ModuleOp>();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  // Naming convention: <original func name>__f<id>. <id> is maintained
  // globally.
  std::string calleeName =
      f.getName().str() + std::string("__f") + std::to_string(id);
  FunctionType calleeType = b.getFunctionType(llvm::None, llvm::None);
  FuncOp callee =
      b.create<FuncOp>(forOps.front()->getLoc(), calleeName, calleeType);

  // Initialize the entry block and the return operation.
  Block *entry = callee.addEntryBlock();
  b.setInsertionPointToStart(entry);
  b.create<mlir::ReturnOp>(callee.getLoc());
  b.setInsertionPointToStart(entry);

  // Grab arguments from the top forOp.
  llvm::SetVector<Value> args;
  getArgs(forOps, args);

  // Argument mapping for cloning. Also intialize arguments to the entry block.
  BlockAndValueMapping mapping;
  for (Value arg : args)
    mapping.map(arg, entry->addArgument(arg.getType()));

  callee.setType(b.getFunctionType(entry->getArgumentTypes(), llvm::None));

  for (Operation *op : forOps)
    b.clone(*op, mapping);

  return {callee, mapping};
}

static CallOp createCaller(MutableArrayRef<Operation *> forOps, FuncOp callee,
                           BlockAndValueMapping vmap, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);

  // Inversed mapping from callee arguments to values in the source function.
  BlockAndValueMapping imap;
  for (auto &it : vmap.getValueMap())
    imap.map(it.second, it.first);

  SmallVector<Value> args;
  transform(callee.getArguments(), std::back_inserter(args),
            [&](Value value) { return imap.lookup(value); });

  // Get function type.
  b.setInsertionPoint(forOps.front());
  CallOp caller = b.create<CallOp>(forOps.front()->getLoc(), callee, args);

  for (Operation *op : forOps)
    op->erase();

  return caller;
}

/// At each loop depth, check if there is any affine loop nest. If there is,
/// stop searching and extract that loop nest into a function.
/// We assume all affine accesses are represented with affine.load/store.
static LogicalResult splitNonAffine(FuncOp f, OpBuilder &b, bool markOnly,
                                    int maxLoopDepth) {
  // First, mark all the loops that have non-affine accesses as
  // scop.non_affine_access
  markNonAffine(f, b);
  if (markOnly)
    return success();

  // Affine loops without non-affine accesses.
  SmallVector<SmallVector<Operation *>> loops;
  std::function<void(int, Block *)> process = [&](int depth, Block *block) {
    if (depth > maxLoopDepth)
      return;

    // Keep consecutive loops.
    SmallVector<Operation *> currLoops;
    for (Operation &op : *block) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        if (forOp->hasAttr("scop.non_affine_access"))
          process(depth + 1, forOp.getBody());
        else {
          if (currLoops.empty() || currLoops.back() == forOp->getPrevNode())
            currLoops.push_back(forOp);
          else {
            loops.push_back(currLoops);
            currLoops = {forOp};
          }
        }
      } else {
        for (Region &region : op.getRegions())
          for (Block &blk : region)
            process(depth, &blk);
      }
    }

    if (!currLoops.empty())
      loops.push_back(currLoops);
  };

  for (Block &block : f.getBody())
    process(0, &block);

  LLVM_DEBUG({
    dbgs() << "Affine loops:\n";
    for (auto &ops : loops) {
      for (Operation *op : ops)
        op->dump();
      dbgs() << "\n\n";
    }
  });

  unsigned startId = 0;
  for (auto &ops : loops) {
    FuncOp callee;
    BlockAndValueMapping vmap;

    std::tie(callee, vmap) =
        createCallee(MutableArrayRef<Operation *>(ops), startId, f, b);
    callee->setAttr("scop.affine", b.getUnitAttr());
    CallOp caller =
        createCaller(MutableArrayRef<Operation *>(ops), callee, vmap, b);
    caller->setAttr("scop.affine", b.getUnitAttr());
  }

  return success();
}

namespace {
struct SplitNonAffinePass : phism::SplitNonAffineBase<SplitNonAffinePass> {
  void runOnFunction() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    // These functions contain only affine loop nests without any non-affine
    // access.
    if (f->hasAttr("scop.affine"))
      return;

    // We should only apply this pass on the function with phism.top.
    if ((!topOnly || f->hasAttr("phism.top")) &&
        failed(splitNonAffine(f, b, markOnly, maxLoopDepth)))
      return signalPassFailure();

    f->setAttr("scop.ignored", b.getUnitAttr());
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
phism::createSplitNonAffinePass() {
  return std::make_unique<SplitNonAffinePass>();
}
