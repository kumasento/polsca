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

#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "loop-extract"

using namespace mlir;
using namespace llvm;
using namespace phism;

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

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // Find if the storeOp has a corresponding load at the same memory address,
    // within the same region.
    SmallVector<AffineStoreOp> worklist;

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

      // storeOp should be the last statement.
      if (!isa<AffineYieldOp>(storeOp->getNextNode()))
        return;

      worklist.push_back(storeOp);
    });

    for (AffineStoreOp storeOp : worklist) {
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
      AffineLoadOp newLoadOp = b.create<AffineLoadOp>(storeOp.getLoc(), newMem,
                                                      storeOp.getAffineMap(),
                                                      storeOp.getMapOperands());
      storeOp.setOperand(0, newLoadOp.getResult());

      // Create a duplicate of the current region.
      mlir::AffineForOp parent = storeOp->getParentOfType<AffineForOp>();
      if (!parent)
        return;

      b.setInsertionPointAfter(parent);
      AffineForOp nextFor = b.create<AffineForOp>(
          parent.getLoc(), parent.getLowerBoundOperands(),
          parent.getLowerBoundMap(), parent.getUpperBoundOperands(),
          parent.getUpperBoundMap());
      b.setInsertionPointToStart(nextFor.getBody());

      // For cloning.
      BlockAndValueMapping vmap;
      vmap.map(parent.getInductionVar(), nextFor.getInductionVar());

      AffineLoadOp clonedLoad = cast<AffineLoadOp>(b.clone(*newLoadOp, vmap));
      vmap.map(newLoadOp.getResult(), clonedLoad.getResult());
      cast<AffineStoreOp>(b.clone(*storeOp, vmap));

      // Finally, remove the old pair.
      storeOp.erase();
      newLoadOp.erase();
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
  callee.setVisibility(SymbolTable::Visibility::Private);

  Block *entry = callee.addEntryBlock();
  b.setInsertionPointToStart(entry);
  b.create<mlir::ReturnOp>(callee.getLoc());
  b.setInsertionPointToStart(entry);

  SetVector<Value> args;
  getArgs(forOp, args);

  BlockAndValueMapping mapping;
  for (Value arg : args)
    mapping.map(arg, entry->addArgument(arg.getType()));
  callee.setType(b.getFunctionType(entry->getArgumentTypes(), llvm::None));
  callee.setVisibility(SymbolTable::Visibility::Public);

  b.clone(*forOp.getOperation(), mapping);

  return {callee, mapping};
}

static void createPointLoopsCaller(AffineForOp startForOp, FuncOp callee,
                                   BlockAndValueMapping vmap, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);

  // Inversed mapping from callee arguments to values in the source function.
  BlockAndValueMapping imap = vmap.getInverse();

  SmallVector<Value> args;
  transform(callee.getArguments(), std::back_inserter(args),
            [&](Value value) { return imap.lookup(value); });

  // Get function type.
  b.setInsertionPoint(startForOp.getOperation());
  b.create<CallOp>(startForOp.getLoc(), callee, args);
  startForOp.erase();
}

static int extractPointLoops(FuncOp f, int startId, OpBuilder &b) {
  ModuleOp m = f->getParentOfType<ModuleOp>();

  SmallVector<Operation *, 4> callers;
  f.walk([&](mlir::CallOp caller) {
    FuncOp callee = m.lookupSymbol<FuncOp>(caller.getCallee());
    if (callee->hasAttr("scop.stmt"))
      callers.push_back(caller);
  });

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  SetVector<Operation *> extracted;

  for (Operation *caller : callers) {
    SmallVector<mlir::AffineForOp, 4> forOps;
    getLoopIVs(*caller, &forOps);

    int pointBandStart = forOps.size();
    while (pointBandStart > 0 && isPointLoop(forOps[pointBandStart - 1])) {
      pointBandStart--;
    }

    // No point loop band.
    if (static_cast<size_t>(pointBandStart) == forOps.size())
      continue;

    mlir::AffineForOp pointBandStartLoop = forOps[pointBandStart];

    // Already visited.
    if (extracted.contains(pointBandStartLoop))
      continue;
    extracted.insert(pointBandStartLoop);

    // Create a callee (function) that wraps the nested loops under the forOp
    // that is the start of a point loops band.
    FuncOp callee;
    BlockAndValueMapping vmap;
    std::tie(callee, vmap) =
        createPointLoopsCallee(pointBandStartLoop, startId, f, b);
    createPointLoopsCaller(pointBandStartLoop, callee, vmap, b);
    startId++;
  }

  return startId;
}

namespace {
struct ExtractPointLoopsPass
    : public mlir::PassWrapper<ExtractPointLoopsPass, OperationPass<ModuleOp>> {
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
      startId += extractPointLoops(f, startId, b);
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

  for (mlir::CallOp caller : callers)
    annotatePointLoops(caller.getOperands(), b);
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

  PassPipelineRegistration<>(
      "loop-transforms", "Phism loop transforms.", [](OpPassManager &pm) {
        pm.addPass(std::make_unique<AnnotatePointLoopsPass>());
        pm.addPass(std::make_unique<ExtractPointLoopsPass>());
        pm.addPass(createCanonicalizerPass());
        // only those private functions will be inlined.
        pm.addPass(createInlinerPass());
      });
}
