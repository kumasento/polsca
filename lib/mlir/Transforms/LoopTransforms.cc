//===- LoopTransforms.cc - Loop transforms ----------------------------C++-===//

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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

#include <queue>

#define DEBUG_TYPE "loop-transforms"

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

static FuncOp getRootFunction(Operation *op) {
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
    FuncOp f = getRootFunction(storeOp);
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

    llvm::SetVector<Operation *> shouldClone;

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

static void getArgs(Operation *parentOp, llvm::SetVector<Value> &args) {
  args.clear();

  llvm::SetVector<Operation *> internalOps;
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
  llvm::SetVector<Value> args;
  getArgs(forOp, args);

  // Argument mapping for cloning. Also intialize arguments to the entry block.
  BlockAndValueMapping mapping;
  for (Value arg : args)
    mapping.map(arg, entry->addArgument(arg.getType()));

  callee.setType(b.getFunctionType(entry->getArgumentTypes(), llvm::None));

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

using LoopTree = llvm::DenseMap<Operation *, llvm::SetVector<Operation *>>;

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
      callee->setAttr("scop.pe", b.getUnitAttr());
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
  llvm::SetVector<Operation *> extracted;

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

/// --------------------- Redistribute statements ---------------------------

static void getAllScopStmts(FuncOp func, llvm::SetVector<FuncOp> &stmts,
                            ModuleOp m) {
  func.walk([&](mlir::CallOp caller) {
    FuncOp callee = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
    if (!callee)
      return;
    if (!callee->hasAttr("scop.stmt"))
      return;

    stmts.insert(callee);
  });
}

static void detectScopPeWithMultipleStmts(ModuleOp m,
                                          llvm::SetVector<mlir::FuncOp> &pes) {
  FuncOp top = getTopFunction(m);
  if (!top)
    return;

  top.walk([&](mlir::CallOp caller) {
    if (!caller->hasAttr("scop.pe"))
      return;

    FuncOp callee = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
    if (!callee)
      return;

    llvm::SetVector<FuncOp> stmts;
    getAllScopStmts(callee, stmts, m);

    if (stmts.size() >= 2)
      pes.insert(callee);
  });
}

static bool hasOnlyReadByScopStmts(FuncOp f, ModuleOp m, Value memref) {
  SmallVector<std::pair<FuncOp, unsigned>> funcAndArgIdx;
  f.walk([&](mlir::CallOp caller) {
    FuncOp callee = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
    if (!callee || !callee->hasAttr("scop.stmt"))
      return;
    auto it = find(caller.getArgOperands(), memref);
    if (it == caller.arg_operand_end())
      return;

    funcAndArgIdx.push_back({callee, it - caller.arg_operand_begin()});
  });

  // Examine the accesses.
  for (auto &it : funcAndArgIdx) {
    FuncOp callee;
    unsigned argIdx;
    std::tie(callee, argIdx) = it;

    assert(callee.getArgument(argIdx).getType().isa<MemRefType>());

    bool hasWriteAccess = false;
    callee.walk([&](mlir::AffineStoreOp storeOp) {
      if (storeOp.getMemRef() == callee.getArgument(argIdx))
        hasWriteAccess = true;
    });

    if (hasWriteAccess)
      return false;
  }

  return true;
}

/// Assuming the memrefs at the top-level are not aliases.
/// Also assuming each scop.stmt will have its accessed memrefs once in its
/// interface.
static bool areScopStmtsSeparable(FuncOp f, ModuleOp m) {
  llvm::SetVector<Value> visited; // memrefs visited.
  llvm::SetVector<Value> conflicted;
  llvm::SetVector<FuncOp> visitedStmts;
  f.walk([&](mlir::CallOp caller) {
    FuncOp callee = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
    if (!callee || !callee->hasAttr("scop.stmt"))
      return;
    if (visitedStmts.count(callee))
      return;
    visitedStmts.insert(callee);

    for (Value arg : caller.getArgOperands())
      if (arg.getType().isa<MemRefType>()) {
        if (visited.count(arg))
          conflicted.insert(arg);
        visited.insert(arg);
      }
  });

  unsigned bad = 0;
  for (auto &memref : conflicted)
    if (!hasOnlyReadByScopStmts(f, m, memref))
      ++bad;

  if (!bad)
    return true;

  LLVM_DEBUG({
    llvm::errs()
        << "\nConflicted memrefs that have not only read accesses:\n\n";
    for (Value memref : conflicted)
      if (!hasOnlyReadByScopStmts(f, m, memref))
        memref.dump();
  });

  return false;
}

/// Erase those affine.for with empty blocks.
static void eraseEmptyAffineFor(FuncOp f) {
  SmallVector<Operation *> eraseOps;
  while (true) {
    eraseOps.clear();
    f.walk([&](mlir::AffineForOp forOp) {
      if (llvm::hasSingleElement(*forOp.getBody())) // the yield
        eraseOps.push_back(forOp.getOperation());
    });
    for (Operation *op : eraseOps)
      op->erase();

    if (eraseOps.empty())
      break;
  }
}

static std::pair<FuncOp, SmallVector<unsigned>>
distributeScopStmt(FuncOp stmt, FuncOp f, ModuleOp m, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfter(f);
  FuncOp newFunc = cast<FuncOp>(b.clone(*f.getOperation()));
  newFunc.setName(std::string(f.getName()) + "__cloned_for__" +
                  std::string(stmt.getName()));

  SmallVector<Operation *> eraseOps;
  newFunc.walk([&](mlir::CallOp caller) {
    if (caller.getCallee() != stmt.getName())
      eraseOps.push_back(caller.getOperation());
  });

  for (Operation *op : eraseOps)
    op->erase();

  eraseEmptyAffineFor(newFunc);

  // Erase not used arguments.
  SmallVector<unsigned> usedArgs;
  for (unsigned i = 0; i < newFunc.getNumArguments(); ++i)
    if (newFunc.getArgument(i).use_empty())
      usedArgs.push_back(i);
  newFunc.eraseArguments(usedArgs);

  return {newFunc, usedArgs};
}

/// The input function will be altered in-place.
static LogicalResult distributeScopStmts(
    FuncOp f, SmallVectorImpl<std::pair<FuncOp, SmallVector<unsigned>>> &dist,
    ModuleOp m, OpBuilder &b) {
  llvm::SetVector<FuncOp> stmts;
  getAllScopStmts(f, stmts, m);

  // Need to duplicate the whole function for each statement. And within each
  // duplication, remove the callers that don't belong there.
  for (FuncOp stmt : stmts) {
    auto res = distributeScopStmt(stmt, f, m, b);
    if (res.first)
      dist.push_back(res);
    else {
      LLVM_DEBUG(dbgs() << "Cannot distribute for: " << stmt.getName() << '\n');
      return failure();
    }
  }

  return success();
}

namespace {
struct RedistributeScopStatementsPass
    : public mlir::PassWrapper<RedistributeScopStatementsPass,
                               OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // -------------------------------------------------------------------
    // Step 1: detect the scop.pe callee that has more than one scop.stmt.
    llvm::SetVector<FuncOp> pes;
    detectScopPeWithMultipleStmts(m, pes);

    if (pes.empty())
      return;

    LLVM_DEBUG({
      llvm::errs() << "-------------------------------------------\n";
      llvm::errs() << "Detected PEs with multiple SCoP statements:\n\n";
      for (FuncOp pe : pes) {
        pe.dump();
        llvm::errs() << "\n------------------------\n\n";
      }
    });

    // -------------------------------------------------------------------
    // Step 2: check if the multiple scop.stmt can be fully separated.
    // The condition is basically each caller refers to different memref.
    /// TODO: carry out alias analysis (not an issue for polybench)
    /// TODO: detailed dependence analysis to cover more cases.
    llvm::SetVector<FuncOp> pesToProc;
    for (FuncOp pe : pes) {
      if (!areScopStmtsSeparable(pe, m)) {
        LLVM_DEBUG({
          llvm::errs() << "Discared " << pe.getName()
                       << "since its scop.stmts are not separable.\n";
        });
        continue;
      }

      pesToProc.insert(pe);
    }

    // -------------------------------------------------------------------
    // Step 3: Process each PE.
    for (FuncOp pe : pesToProc) {
      SmallVector<std::pair<FuncOp, SmallVector<unsigned>>> dists;
      if (failed(distributeScopStmts(pe, dists, m, b))) {
        LLVM_DEBUG({
          llvm::errs() << "Failed to distribute scop.stmt: " << pe.getName()
                       << "\n";
        });
        continue;
      }

      SmallVector<mlir::CallOp> callers;
      m.walk([&](mlir::CallOp caller) {
        if (caller.getCallee() == pe.getName())
          callers.push_back(caller);
      });

      for (mlir::CallOp caller : callers) {
        b.setInsertionPointAfter(caller);
        for (auto dist : dists) {
          FuncOp callee;
          SmallVector<unsigned> erased;
          std::tie(callee, erased) = dist;

          SmallVector<Value> operands;
          for (auto arg : enumerate(caller.getOperands()))
            if (find(erased, arg.index()) == erased.end())
              operands.push_back(arg.value());

          mlir::CallOp newCaller =
              b.create<CallOp>(caller.getLoc(), callee, operands);
          newCaller->setAttr("scop.pe", b.getUnitAttr());
        }
      }

      for (mlir::CallOp caller : callers)
        caller.erase();
      pe.erase();
    }
  }
};
} // namespace

/// --------------------- Loop merge pass ---------------------------

static LogicalResult loopMergeOnScopStmt(FuncOp f, ModuleOp m, OpBuilder &b) {
  llvm::SetVector<FuncOp> stmts;
  getAllScopStmts(f, stmts, m);

  if (!llvm::hasSingleElement(stmts)) {
    LLVM_DEBUG(
        dbgs()
        << "Being conservative not to merge loops with multiple scop.stmts.\n");
    return failure();
  }

  FuncOp targetStmt = *stmts.begin();

  // Get all the callers for the target scop.stmt
  SmallVector<mlir::CallOp> callers;
  f.walk([&](mlir::CallOp caller) {
    if (caller.getCallee() == targetStmt.getName())
      callers.push_back(caller);
  });

  if (hasSingleElement(callers)) {
    LLVM_DEBUG(dbgs() << "There is only one caller instance for PE: "
                      << f.getName() << ".\n");
    return failure();
  }

  // ----------------------------------------------------------------------
  // Step 1: make sure there are no empty sets in loop domains.
  llvm::SetVector<Operation *> erased;
  for (mlir::CallOp caller : callers) {
    SmallVector<Operation *> ops;
    getEnclosingAffineForAndIfOps(*caller.getOperation(), &ops);

    FlatAffineValueConstraints cst;
    assert(succeeded(getIndexSet(ops, &cst)));

    if (!cst.findIntegerSample().hasValue()) {
      LLVM_DEBUG({
        dbgs() << "Found a caller in an empty loop nest.\n";
        caller.dump();
      });
      erased.insert(caller.getOperation());
    };
  }

  callers.erase(remove_if(callers,
                          [&](mlir::CallOp caller) {
                            return erased.count(caller.getOperation());
                          }),
                callers.end());
  for (Operation *op : erased)
    op->erase();

  eraseEmptyAffineFor(f);

  if (hasSingleElement(callers)) {
    LLVM_DEBUG(dbgs() << "There is only one caller instance for PE: "
                      << f.getName() << " after empty loop removal.\n");
    return failure();
  }

  // ----------------------------------------------------------------------
  // Step 2: gather loop structure
  // Make sure the callers have the same prefix, only the last forOp different.
  SmallVector<mlir::AffineForOp> outerLoops;
  SmallVector<mlir::AffineForOp> innermosts; // each corresponds to a caller.
  for (mlir::CallOp caller : callers) {
    SmallVector<Operation *> ops;
    getEnclosingAffineForAndIfOps(*caller.getOperation(), &ops);

    if (ops.empty()) {
      LLVM_DEBUG(dbgs() << "Callers should be wrapped within loops.\n");
      return failure();
    }

    if (any_of(ops, [&](Operation *op) { return isa<mlir::AffineIfOp>(op); })) {
      LLVM_DEBUG(dbgs() << "Cannot deal with affine.if yet.\n");
      return failure();
    }

    // Initialise
    if (outerLoops.empty()) {
      innermosts.push_back(cast<mlir::AffineForOp>(ops.back()));
      ops.pop_back();

      for (Operation *op : ops)
        outerLoops.push_back(cast<mlir::AffineForOp>(op));
    } else {
      SmallVector<mlir::AffineForOp> tmpOuters;
      mlir::AffineForOp innermost;

      innermost = cast<mlir::AffineForOp>(ops.back());
      ops.pop_back();

      for (Operation *op : ops)
        tmpOuters.push_back(cast<mlir::AffineForOp>(op));

      if (tmpOuters != outerLoops) {
        LLVM_DEBUG(dbgs() << "Outer loops are not the same among statements "
                             "(given the last being different).\n");
        return failure();
      }

      if (find(innermosts, innermost) != innermosts.end()) {
        LLVM_DEBUG(dbgs() << "Weird to find the same loop structures between "
                             "two caller instances.\n");
        return failure();
      }

      innermosts.push_back(innermost);
    }
  }

  LLVM_DEBUG({
    dbgs() << "\n-----------------------------------\n";
    dbgs() << "Merging PE: \n";
    f.dump();
  });

  // ----------------------------------------------------------------------
  // Step 3: Affine analysis
  // Check if the innermost loops have no intersection.
  SmallVector<FlatAffineValueConstraints, 4> csts;
  transform(innermosts, std::back_inserter(csts), [&](mlir::AffineForOp forOp) {
    FlatAffineValueConstraints cst;
    cst.addInductionVarOrTerminalSymbol(forOp.getInductionVar());

    LLVM_DEBUG(cst.dump());

    return cst;
  });

  // Make every constraint has the same induction variable.
  for (unsigned i = 1; i < csts.size(); ++i)
    csts[i].setValue(0, csts[0].getValue(0));

  // Check if all the constraints share the same number of columns.
  for (unsigned i = 1; i < csts.size(); ++i) {
    if (csts[i].getNumCols() != csts[0].getNumCols()) {
      LLVM_DEBUG(dbgs() << "Number of columns don't match between two "
                           "candidate constraints.\n");
      return failure();
    }
  }

  // Check if two loops have intersection.
  for (unsigned i = 0; i < csts.size(); ++i)
    for (unsigned j = i + 1; j < csts.size(); ++j) {
      FlatAffineValueConstraints tmp{csts[i]};
      tmp.append(csts[j]);

      if (tmp.findIntegerSample().hasValue()) {
        LLVM_DEBUG(dbgs() << "There is intersection between two innermost "
                             "loops. Cannot merge them safely.\n");
        return failure();
      }
    }

  // Merge: check if one can be merged into another iteratively, until there is
  // no chance of merging.
  while (true) {
    bool merged = false;

    mlir::AffineForOp loopToErase;

    for (unsigned i = 0; i < innermosts.size() && !merged; ++i)
      for (unsigned j = 0; j < innermosts.size() && !merged; ++j) {
        if (i == j)
          continue;

        mlir::AffineForOp loop1 = innermosts[i];
        mlir::AffineForOp loop2 = innermosts[j];

        AffineMap ubMap = loop1.getUpperBoundMap();

        // Condition BEGIN -
        if (loop2.getLowerBoundMap().isSingleConstant()) {
          int64_t constLb = loop2.getLowerBoundMap().getSingleConstantResult();
          for (AffineExpr ub : ubMap.getResults()) {
            if (AffineConstantExpr constUbExpr =
                    ub.dyn_cast<AffineConstantExpr>()) {
              int64_t constUb = constUbExpr.getValue();
              if (constLb == constUb) {
                // Condition END -
                LLVM_DEBUG(dbgs()
                           << "Found loop2's single constant lower bound "
                           << constLb
                           << " equals to one of the upper bounds of loop1 "
                           << constUb
                           << ". We can merge them together since loop1 and "
                              "loop2 don't intersect.\n");

                merged = true;

                // Set to erase;
                loopToErase = loop2;

                // Set the new upper bound;
                llvm::SetVector<AffineExpr> results;
                for (AffineExpr expr : ubMap.getResults())
                  if (expr != ub)
                    results.insert(expr);
                for (AffineExpr expr : loop2.getUpperBoundMap().getResults())
                  results.insert(expr);

                AffineMap newUbMap =
                    AffineMap::get(ubMap.getNumDims(), ubMap.getNumSymbols(),
                                   results.takeVector(), ubMap.getContext());
                LLVM_DEBUG({
                  dbgs() << "New upper bound: \n";
                  newUbMap.dump();
                });
                loop1.setUpperBoundMap(newUbMap);

                break;
              }
            }
          }
        }
      }

    if (loopToErase) {
      innermosts.erase(find(innermosts, loopToErase));
      loopToErase.erase();
    }

    if (!merged)
      break;
  }

  return success();
}

namespace {

/// Will only work within scop.pe on scop.stmt to avoid side effects.
struct LoopMergePass
    : public mlir::PassWrapper<LoopMergePass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<FuncOp> pes;
    FuncOp f = getTopFunction(m);
    if (!f)
      return;

    f.walk([&](mlir::CallOp caller) {
      if (!caller->hasAttr("scop.pe"))
        return;
      FuncOp pe = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
      if (!pe)
        return;
      pes.push_back(pe);
    });

    for (FuncOp pe : pes) {
      if (failed(loopMergeOnScopStmt(pe, m, b))) {
        LLVM_DEBUG(dbgs() << "Failed to merge loops in: " << pe.getName()
                          << ".\n");
      }
    }
  }
};

} // namespace

/// -------------------------- Scop stmt inline -------------------------------

static LogicalResult inlineScopStmtWithinFunction(FuncOp f, FuncOp stmt,
                                                  OpBuilder &b) {
  if (f->hasAttr("scop.stmt")) // skipped.
    return success();

  SmallVector<mlir::CallOp> callers;
  f.walk([&](mlir::CallOp caller) {
    if (caller.getCallee() == stmt.getName())
      callers.push_back(caller);
  });

  // Replace each caller with the statement body.
  for (mlir::CallOp caller : callers) {
    b.setInsertionPointAfter(caller);

    BlockAndValueMapping vmap;
    vmap.map(stmt.getArguments(), caller.getArgOperands());

    // We know that the body of the stmt is simply a list of operations without
    // region.
    for (Operation &op : stmt.getBlocks().begin()->getOperations())
      if (!isa<mlir::ReturnOp>(op))
        b.clone(op, vmap);
  }

  // Erase the callers.
  for (mlir::CallOp caller : callers)
    caller.erase();

  return success();
}

namespace {

/// Try to merge all the functions with attribute {scop.stmt}.
struct ScopStmtInlinePass
    : public mlir::PassWrapper<ScopStmtInlinePass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<FuncOp> stmts;
    SmallVector<FuncOp> funcs;

    m.walk([&](FuncOp f) {
      if (f->hasAttr("scop.stmt"))
        stmts.push_back(f);
      else
        funcs.push_back(f);
    });

    // We know that a scop.stmt won't call another scop.stmt.
    for (FuncOp stmt : stmts) {
      bool hasCaller = false;
      stmt.walk([&](mlir::CallOp caller) { hasCaller = true; });

      assert(!hasCaller && "A scop.stmt cannot call another function.");
    }

    // Iterate every scop.stmt that should be inlined.
    for (FuncOp stmt : stmts) {
      for (FuncOp func : funcs)
        if (failed(inlineScopStmtWithinFunction(func, stmt, b)))
          return;
      stmt.erase();
    }
  }
};

} // namespace

/// ------------------ Loop coalescing --------------------------

static void getLoopBand(mlir::AffineForOp topOp,
                        SmallVectorImpl<mlir::AffineForOp> &band) {
  Operation *currOp = topOp;
  mlir::AffineForOp forOp;
  while ((forOp = dyn_cast<mlir::AffineForOp>(currOp))) {
    band.push_back(forOp);

    auto &ops = forOp.getBody()->getOperations();
    if (ops.size() != 2)
      break;
    currOp = &ops.front();
  }
}

static void demoteBoundToIf(mlir::AffineForOp forOp,
                            llvm::SetVector<Value> &ivs, OpBuilder &b) {
  Location loc = forOp.getLoc();

  auto lbMap = forOp.getLowerBoundMap();
  auto lbOperands = forOp.getLowerBoundOperands();
  auto ubMap = forOp.getUpperBoundMap();
  auto ubOperands = forOp.getUpperBoundOperands();

  // If none of the operands is a loop induction variable, there is no need to
  // continue;
  if (none_of(lbOperands, [&](Value value) { return ivs.count(value) != 0; }) &&
      none_of(ubOperands, [&](Value value) { return ivs.count(value) != 0; }))
    return;

  // Gather the constraints from the lbMap that involve the outer IVs.
  auto indicesOfIV = [&](OperandRange operands) -> auto {
    llvm::SetVector<int64_t> indices;
    for (auto operand : enumerate(lbOperands))
      if (ivs.count(operand.value()))
        indices.insert(operand.index());
    return indices;
  };

  auto lbIndices = indicesOfIV(lbOperands);
  auto ubIndices = indicesOfIV(ubOperands);

  // Gather affine expressions from the affine map that involves these indices.
  auto boundsToDemote = [&](llvm::SetVector<int64_t> indices,
                            AffineMap affMap) -> auto {
    SmallVector<AffineExpr> demoted, remained;

    for (AffineExpr expr : affMap.getResults()) {
      bool toDemote = false;
      expr.walk([&](AffineExpr e) {
        if ((e.isa<AffineDimExpr>() || e.isa<AffineSymbolExpr>()) &&
            indices.count(e.isa<AffineDimExpr>()
                              ? e.cast<AffineDimExpr>().getPosition()
                              : (e.cast<AffineSymbolExpr>().getPosition() +
                                 affMap.getNumDims())))
          toDemote = true;
      });

      if (toDemote)
        demoted.push_back(expr);
      else
        remained.push_back(expr);
    }

    return std::make_pair(demoted, remained);
  };

  SmallVector<AffineExpr> lbBoundsToDemote, lbBoundsToRemain;
  SmallVector<AffineExpr> ubBoundsToDemote, ubBoundsToRemain;

  std::tie(lbBoundsToDemote, lbBoundsToRemain) =
      boundsToDemote(lbIndices, lbMap);
  std::tie(ubBoundsToDemote, ubBoundsToRemain) =
      boundsToDemote(ubIndices, ubMap);

  LLVM_DEBUG({
    dbgs() << "Lower bounds to demote:\n";
    for (AffineExpr expr : lbBoundsToDemote)
      dbgs() << expr << '\n';
    dbgs() << "Upper bounds to demote:\n";
    for (AffineExpr expr : ubBoundsToDemote)
      dbgs() << expr << '\n';
  });

  // The new affine.if should constrain the IV of the current loop to be >=
  // lbBoundsToDemote and <= ubBoundsToDemote.
  // Need to make sure the dims/symbols are aligned.

  auto boundConstraints = [&](OperandRange operands, AffineMap affMap,
                              ArrayRef<AffineExpr> boundsToDemote,
                              bool isLowerBound =
                                  false) -> FlatAffineValueConstraints {
    SmallVector<Value> values(operands);

    FlatAffineValueConstraints cst(affMap.getNumDims() + 1,
                                   affMap.getNumSymbols(), 0UL);
    for (unsigned i = 0; i < affMap.getNumDims(); ++i)
      cst.setValue(i, values[i]);
    cst.setValue(affMap.getNumDims(), forOp.getInductionVar());
    for (unsigned i = 0; i < affMap.getNumSymbols(); ++i)
      cst.setValue(i + affMap.getNumDims() + 1,
                   values[i + affMap.getNumDims()]);

    assert(succeeded(cst.addBound(
        isLowerBound ? FlatAffineValueConstraints::BoundType::LB
                     : FlatAffineValueConstraints::BoundType::UB,
        affMap.getNumDims(),
        AffineMap::get(affMap.getNumDims() + 1, affMap.getNumSymbols(),
                       boundsToDemote, b.getContext()))));
    return cst;
  };

  auto lbBoundCst = boundConstraints(lbOperands, lbMap, lbBoundsToDemote,
                                     /*isLowerBound=*/true);
  auto ubBoundCst = boundConstraints(ubOperands, ubMap, ubBoundsToDemote,
                                     /*isLowerBound=*/false);

  lbBoundCst.mergeAndAlignIdsWithOther(0, &ubBoundCst);
  lbBoundCst.append(ubBoundCst);

  LLVM_DEBUG({
    dbgs() << "Created constraints for the new affine.if:\n";
    lbBoundCst.dump();
  });

  b.setInsertionPointToStart(forOp.getBody());

  SmallVector<Value> ifOperands;
  lbBoundCst.getAllValues(&ifOperands);
  auto ifOp = b.create<mlir::AffineIfOp>(
      loc, lbBoundCst.getAsIntegerSet(b.getContext()), ifOperands,
      /*withElseRegion=*/false);

  SmallVector<Operation *> toErase;
  b.setInsertionPointToStart(ifOp.getThenBlock());
  BlockAndValueMapping vmap;
  for (Operation &op : forOp.getBody()->getOperations())
    if (&op != ifOp && !isa<mlir::AffineYieldOp>(&op)) {
      b.clone(op, vmap);
      toErase.push_back(&op);
    }

  for (Operation *op : reverse(toErase)) {
    LLVM_DEBUG(dbgs() << "Erasing " << *op << '\n');
    op->erase();
  }

  // Update the for loop bounds.
  forOp.setLowerBoundMap(AffineMap::get(lbMap.getNumDims(),
                                        lbMap.getNumSymbols(), lbBoundsToRemain,
                                        b.getContext()));
  forOp.setUpperBoundMap(AffineMap::get(ubMap.getNumDims(),
                                        ubMap.getNumSymbols(), ubBoundsToRemain,
                                        b.getContext()));
}

static void demoteBoundToIf(FuncOp f, OpBuilder &b) {
  for (Operation &op : f.getBody().getOps()) {
    if (auto forOp = dyn_cast<mlir::AffineForOp>(op)) {
      // Get loop band.
      SmallVector<mlir::AffineForOp> band;
      getLoopBand(forOp, band);

      if (band.size() <= 1)
        continue;

      LLVM_DEBUG(forOp.dump());
      LLVM_DEBUG(dbgs() << "Found loop band of size: " << band.size() << '\n');

      // Get all loop IVs.
      llvm::SetVector<Value> ivs;
      for (mlir::AffineForOp op : band)
        ivs.insert(op.getInductionVar());

      // Find if an inner loop has a bound that depends on an outer loop IV.
      for (mlir::AffineForOp op : band)
        if (op != forOp) // skip the top
          demoteBoundToIf(op, ivs, b);
    }
  }
}

namespace {
struct DemoteBoundToIfPass
    : public PassWrapper<DemoteBoundToIfPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<FuncOp> worklist;
    m.walk([&](FuncOp f) {
      if (f->hasAttr("scop.pe"))
        worklist.push_back(f);
    });

    for (FuncOp f : worklist)
      demoteBoundToIf(f, b);
  }
};
} // namespace

void phism::registerLoopTransformPasses() {
  // PassRegistration<AnnotatePointLoopsPass>(
  //     "annotate-point-loops", "Annotate loops with point/tile info.");
  // PassRegistration<ExtractPointLoopsPass>(
  //     "extract-point-loops", "Extract point loop bands into functions");
  PassPipelineRegistration<>(
      "redis-scop-stmts",
      "Redistribute scop statements across extracted point loops.",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<RedistributeScopStatementsPass>());
      });
  PassPipelineRegistration<>(
      "loop-merge", "Merge loops by affine analysis.",
      [](OpPassManager &pm) { pm.addPass(std::make_unique<LoopMergePass>()); });

  PassPipelineRegistration<>(
      "improve-pipelining", "Improve the pipelining performance",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<InsertScratchpadPass>());
      });

  PassPipelineRegistration<>(
      "scop-stmt-inline", "Inline scop.stmt", [](OpPassManager &pm) {
        pm.addPass(std::make_unique<ScopStmtInlinePass>());
        pm.addPass(createCanonicalizerPass());
      });

  PassPipelineRegistration<LoopTransformsPipelineOptions>(
      "loop-transforms", "Phism loop transforms.",
      [](OpPassManager &pm,
         const LoopTransformsPipelineOptions &pipelineOptions) {
        pm.addPass(std::make_unique<AnnotatePointLoopsPass>());
        pm.addPass(std::make_unique<ExtractPointLoopsPass>(pipelineOptions));
        pm.addPass(createCanonicalizerPass());
      });

  PassPipelineRegistration<>(
      "loop-redis-and-merge", "Redistribute stmts and merge loops.",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<RedistributeScopStatementsPass>());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(std::make_unique<LoopMergePass>());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(std::make_unique<ScopStmtInlinePass>());
        pm.addPass(createCanonicalizerPass());
      });

  PassPipelineRegistration<>(
      "demote-bound-to-if",
      "Demote bounds on outer loop induction variables to affine.if.",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<DemoteBoundToIfPass>());
        pm.addPass(createCanonicalizerPass());
      });
}
