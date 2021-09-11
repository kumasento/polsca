//===- LoopTransforms.cc - Loop transforms ----------------------------C++-===//

#include "phism/mlir/Transforms/PhismTransforms.h"

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

#include <queue>

#define DEBUG_TYPE "loop-extract"

using namespace mlir;
using namespace llvm;
using namespace phism;

namespace {
struct LoopTransformsPipelineOptions
    : public mlir::PassPipelineOptions<LoopTransformsPipelineOptions> {

  Option<int> maxSpan{*this, "max-span",
                      llvm::cl::desc("Maximum spanning of the point loop.")};
};

} // namespace

/// -------------------------- Insert Scratchpad ---------------------------

static FuncOp getTopFunction(Operation *op) {
  while (!op->getParentOfType<FuncOp>())
    op = op->getParentOp();
  return op->getParentOfType<FuncOp>();
}

namespace {

///
struct InsertScratchpadPass
    : public PassWrapper<InsertScratchpadPass, OperationPass<ModuleOp>> {

  void process(AffineStoreOp storeOp, ModuleOp m, OpBuilder &b) {
    DominanceInfo dom(storeOp->getParentOp());
    Value mem = storeOp.getMemRef();
    // TODO: we should further check the address being accessed.
    FuncOp f = getTopFunction(storeOp);
    b.setInsertionPointToStart(&f.getBlocks().front());

    // New scratchpad memory
    // TODO: reduce its size to fit the iteration domain.
    memref::AllocaOp newMem = b.create<memref::AllocaOp>(
        storeOp.getLoc(), mem.getType().cast<MemRefType>());

    // Add a new store to the scratchpad.
    b.setInsertionPoint(storeOp);
    Operation *newStoreOp = b.clone(*storeOp.getOperation());
    cast<AffineStoreOp>(newStoreOp).setMemRef(newMem);

    // Load from the scratchpad and store to the original address.
    b.setInsertionPoint(storeOp);
    storeOp.getAffineMap().dump();
    AffineLoadOp newLoadOp =
        b.create<AffineLoadOp>(storeOp.getLoc(), newMem, storeOp.getAffineMap(),
                               storeOp.getMapOperands());
    // Replace the loaded result for all the future uses.
    storeOp.getValueToStore().replaceUsesWithIf(
        newLoadOp.getResult(), [&](OpOperand &operand) {
          return dom.dominates(newLoadOp.getOperation(), operand.getOwner());
        });

    // Create a duplicate of the current region.
    mlir::AffineForOp parent = storeOp->getParentOfType<AffineForOp>();
    if (!parent)
      return;

    // Split the block.
    Block *prevBlock = parent.getBody();
    Block *nextBlock = prevBlock->splitBlock(newLoadOp.getOperation());

    // Post-fix the prev block with the missed termination op.
    b.setInsertionPointToEnd(prevBlock);
    b.create<AffineYieldOp>(storeOp.getLoc());

    // Create a new for cloned after the parent.
    b.setInsertionPointAfter(parent);
    AffineForOp nextFor = b.create<AffineForOp>(
        parent.getLoc(), parent.getLowerBoundOperands(),
        parent.getLowerBoundMap(), parent.getUpperBoundOperands(),
        parent.getUpperBoundMap());

    // Clone every operation from the next block into the new for loop.
    BlockAndValueMapping vmap;
    vmap.map(parent.getInductionVar(), nextFor.getInductionVar());

    SetVector<Operation *> shouldClone;

    b.setInsertionPointToStart(nextFor.getBody());
    for (Operation &op : nextBlock->getOperations())
      if (!isa<AffineYieldOp>(op)) {
        Operation *cloned = b.clone(op, vmap);

        // If any operand from the cloned operator is defined from the original
        // region, we should clone them as well.
        for (auto operand : cloned->getOperands())
          if (operand.getParentRegion() == op.getParentRegion())
            shouldClone.insert(operand.getDefiningOp());
      }

    b.setInsertionPointToStart(nextFor.getBody());
    for (Operation &op : prevBlock->getOperations())
      if (shouldClone.count(&op)) {
        Operation *cloned = b.clone(op, vmap);
        for (unsigned i = 0; i < cloned->getNumResults(); ++i)
          op.getResult(i).replaceUsesWithIf(
              cloned->getResult(i), [&](OpOperand &operand) {
                return operand.getOwner()->getBlock() == nextFor.getBody();
              });
      }

    // Clean up
    nextBlock->erase();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    while (true) {
      AffineStoreOp storeOpToProcess = nullptr; // To process.

      // Find the store op.
      m.walk([&](AffineStoreOp storeOp) {
        DominanceInfo dom(storeOp->getParentOp());

        SmallVector<Operation *> users;

        Value mem = storeOp.getMemRef();
        for (Operation *user : mem.getUsers())
          if (user->getParentRegion() == storeOp->getParentRegion())
            users.push_back(user);

        // Store is the only user, no need to insert scratchpad.
        if (users.size() == 1)
          return;

        unsigned numStoreOps = 0;
        for (Operation *user : users)
          if (isa<AffineStoreOp>(user))
            numStoreOps++;

        // We only deal with the case that there is only one write.
        if (numStoreOps > 1)
          return;

        // Check if all load operations are dominating the store.
        for (Operation *user : users)
          if (isa<AffineLoadOp>(user) && !dom.dominates(user, storeOp))
            return;

        storeOpToProcess = storeOp;
      });

      if (!storeOpToProcess)
        break;

      process(storeOpToProcess, m, b);
    }
  }
};
} // namespace

/// -------------------------- Extract point loops ---------------------------

/// Check if the provided function has point loops in it.
static bool hasPointLoops(FuncOp f) {
  bool hasPointLoop = false;
  f.walk([&](mlir::AffineForOp forOp) {
    if (!hasPointLoop)
      hasPointLoop = forOp->hasAttr("scop.point_loop");
  });
  return hasPointLoop;
}

static bool isPointLoop(mlir::AffineForOp forOp) {
  return forOp->hasAttr("scop.point_loop");
}

static void getArgs(Operation *parentOp, SetVector<Value> &args) {
  args.clear();

  SetVector<Operation *> internalOps;
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

static std::pair<FuncOp, BlockAndValueMapping>
createPointLoopsCallee(mlir::AffineForOp forOp, int id, FuncOp f,
                       OpBuilder &b) {
  ModuleOp m = f->getParentOfType<ModuleOp>();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  // Naming convention: <original func name>__PE<id>. <id> is maintained
  // globally.
  std::string calleeName =
      f.getName().str() + std::string("__PE") + std::to_string(id);
  FunctionType calleeType = b.getFunctionType(llvm::None, llvm::None);
  FuncOp callee = b.create<FuncOp>(forOp.getLoc(), calleeName, calleeType);

  // Initialize the entry block and the return operation.
  Block *entry = callee.addEntryBlock();
  b.setInsertionPointToStart(entry);
  b.create<mlir::ReturnOp>(callee.getLoc());
  b.setInsertionPointToStart(entry);

  // Grab arguments from the top forOp.
  SetVector<Value> args;
  getArgs(forOp, args);

  // Argument mapping for cloning. Also intialize arguments to the entry block.
  BlockAndValueMapping mapping;
  for (Value arg : args)
    mapping.map(arg, entry->addArgument(arg.getType()));

  callee.setType(b.getFunctionType(entry->getArgumentTypes(), llvm::None));
  callee.setVisibility(SymbolTable::Visibility::Public);

  b.clone(*forOp.getOperation(), mapping);

  return {callee, mapping};
}

static CallOp createPointLoopsCaller(AffineForOp startForOp, FuncOp callee,
                                     BlockAndValueMapping vmap, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);

  // Inversed mapping from callee arguments to values in the source function.
  BlockAndValueMapping imap = vmap.getInverse();

  SmallVector<Value> args;
  transform(callee.getArguments(), std::back_inserter(args),
            [&](Value value) { return imap.lookup(value); });

  // Get function type.
  b.setInsertionPoint(startForOp.getOperation());
  CallOp caller = b.create<CallOp>(startForOp.getLoc(), callee, args);
  startForOp.erase();
  return caller;
}

using LoopTree = llvm::DenseMap<Operation *, SetVector<Operation *>>;

/// Returns true if itself or any of the descendants has been extracted.
static bool greedyLoopExtraction(Operation *op, const int maxSpan, int &startId,
                                 LoopTree &loopTree, FuncOp &f, OpBuilder &b) {
  if (!loopTree.count(op)) // there is no descendant.
    return false;

  // If op is the root node or given that maxSpan has been specified, there are
  // more children than that number, then we should extract all the children
  // into functions.
  bool shouldExtract =
      isa<FuncOp>(op) || (maxSpan > 0 && (int)loopTree[op].size() > maxSpan);

  SmallPtrSet<Operation *, 4> extracted;
  for (Operation *child : loopTree[op])
    if (greedyLoopExtraction(child, maxSpan, startId, loopTree, f, b))
      extracted.insert(child);

  // If there are any child has been extracted, this whole subtree should be
  // extracted.
  shouldExtract |= !extracted.empty();

  if (shouldExtract)
    for (Operation *child : loopTree[op]) {
      // Don't extract again.
      if (extracted.count(child))
        continue;

      mlir::AffineForOp forOp = cast<mlir::AffineForOp>(child);
      assert(forOp->hasAttr("scop.point_loop") &&
             "The forOp to be extracted should be a point loop.");

      FuncOp callee;
      BlockAndValueMapping vmap;

      std::tie(callee, vmap) = createPointLoopsCallee(forOp, startId, f, b);
      CallOp caller = createPointLoopsCaller(forOp, callee, vmap, b);
      caller->setAttr("scop.pe", b.getUnitAttr());

      startId++;
    }

  return shouldExtract;
}

static int extractPointLoops(FuncOp f, int startId, int maxSpan, OpBuilder &b) {
  ModuleOp m = f->getParentOfType<ModuleOp>();

  // Get the scop.stmt callers. These are what we focus on.
  SmallVector<Operation *, 4> callers;
  f.walk([&](mlir::CallOp caller) {
    FuncOp callee = m.lookupSymbol<FuncOp>(caller.getCallee());
    if (callee->hasAttr("scop.stmt"))
      callers.push_back(caller);
  });

  // Place to insert the new function.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  // Those point loops that has been visited and extracted.
  SetVector<Operation *> extracted;

  // Map from a point loop to its children.
  LoopTree loopTree;
  loopTree[f] = {}; // root node is the function.

  // Build the tree.
  for (Operation *caller : callers) {
    SmallVector<mlir::AffineForOp> forOps;
    getLoopIVs(*caller, &forOps);

    mlir::AffineForOp lastPointLoop = nullptr;
    std::reverse(forOps.begin(), forOps.end());
    for (unsigned i = 0; i < forOps.size(); i++) {
      if (!isPointLoop(forOps[i]))
        break;

      if (i == 0 && !loopTree.count(forOps[i]))
        loopTree[forOps[i]] = {}; // leaf
      if (i > 0)
        loopTree[forOps[i]].insert(forOps[i - 1]);
      lastPointLoop = forOps[i];
    }

    if (lastPointLoop)
      loopTree[f].insert(lastPointLoop);
  }

  greedyLoopExtraction(f, maxSpan, startId, loopTree, f, b);

  return startId;
}

namespace {
struct ExtractPointLoopsPass
    : public mlir::PassWrapper<ExtractPointLoopsPass, OperationPass<ModuleOp>> {

  int maxSpan = -1;

  ExtractPointLoopsPass() = default;
  ExtractPointLoopsPass(const ExtractPointLoopsPass &pass) {}
  ExtractPointLoopsPass(const LoopTransformsPipelineOptions &options)
      : maxSpan(!options.maxSpan.hasValue() ? -1 : options.maxSpan.getValue()) {
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<FuncOp, 4> fs;
    m.walk([&](FuncOp f) {
      if (hasPointLoops(f))
        fs.push_back(f);
    });

    int startId = 0;
    for (FuncOp f : fs)
      startId += extractPointLoops(f, startId, maxSpan, b);
  }
};
} // namespace

/// -------------------------- Annotate point loops ---------------------------

/// A recursive function. Terminates when all operands are not defined by
/// affine.apply, nor loop IVs.
static void annotatePointLoops(ValueRange operands, OpBuilder &b) {
  for (mlir::Value operand : operands) {
    // If a loop IV is directly passed into the statement call.
    if (BlockArgument arg = operand.dyn_cast<BlockArgument>()) {
      mlir::AffineForOp forOp =
          dyn_cast<mlir::AffineForOp>(arg.getOwner()->getParentOp());
      if (forOp) {
        // An affine.for that has its indunction var used by a scop.stmt
        // caller is a point loop.
        forOp->setAttr("scop.point_loop", b.getUnitAttr());
      }
    } else {
      mlir::AffineApplyOp applyOp =
          operand.getDefiningOp<mlir::AffineApplyOp>();
      if (applyOp) {
        // Mark the parents of its operands, if a loop IVs, as point loops.
        annotatePointLoops(applyOp.getOperands(), b);
      }
    }
  }
}

/// Annotate loops in the dst to indicate whether they are point/tile loops.
/// Should only call this after -canonicalize.
/// TODO: Support handling index calculation, e.g., jacobi-1d.
static void annotatePointLoops(FuncOp f, OpBuilder &b) {
  ModuleOp m = f->getParentOfType<ModuleOp>();
  assert(m && "A FuncOp should be wrapped in a ModuleOp");

  SmallVector<mlir::CallOp> callers;
  f.walk([&](mlir::CallOp caller) {
    FuncOp callee = m.lookupSymbol<FuncOp>(caller.getCallee());
    assert(callee && "Callers should have its callees available.");

    // Only gather callers that calls scop.stmt
    if (callee->hasAttr("scop.stmt"))
      callers.push_back(caller);
  });

  for (mlir::CallOp caller : callers) {
    annotatePointLoops(caller.getOperands(), b);

    // Post-fix intermediate forOps.
    SmallVector<mlir::AffineForOp, 4> forOps;
    getLoopIVs(*caller.getOperation(), &forOps);

    bool started = false;
    for (mlir::AffineForOp forOp : forOps) {
      if (forOp->hasAttr("scop.point_loop"))
        started = true;
      else if (started)
        forOp->setAttr("scop.point_loop", b.getUnitAttr());
    }
  }
}

namespace {
struct AnnotatePointLoopsPass
    : public mlir::PassWrapper<AnnotatePointLoopsPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    annotatePointLoops(f, b);
  }
};
} // namespace

void phism::registerLoopTransformPasses() {
  PassRegistration<AnnotatePointLoopsPass>(
      "annotate-point-loops", "Annotate loops with point/tile info.");
  PassRegistration<ExtractPointLoopsPass>(
      "extract-point-loops", "Extract point loop bands into functions");

  PassPipelineRegistration<>(
      "improve-pipelining", "Improve the pipelining performance",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<InsertScratchpadPass>());
      });

  PassPipelineRegistration<LoopTransformsPipelineOptions>(
      "loop-transforms", "Phism loop transforms.",
      [](OpPassManager &pm,
         const LoopTransformsPipelineOptions &pipelineOptions) {
        pm.addPass(std::make_unique<AnnotatePointLoopsPass>());
        pm.addPass(std::make_unique<ExtractPointLoopsPass>(pipelineOptions));
        pm.addPass(createCanonicalizerPass());
        // only those private functions will be inlined.
        pm.addPass(createInlinerPass());
      });
}
