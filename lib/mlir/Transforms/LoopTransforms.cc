//===- LoopTransforms.cc - Loop transforms ----------------------------C++-===//

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

/// --------------------- Redistribute statements ---------------------------

static void getAllScopStmts(FuncOp func, SetVector<FuncOp> &stmts, ModuleOp m) {
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
                                          SetVector<mlir::FuncOp> &pes) {
  FuncOp top = getTopFunction(m);
  top.walk([&](mlir::CallOp caller) {
    if (!caller->hasAttr("scop.pe"))
      return;

    FuncOp callee = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
    if (!callee)
      return;

    SetVector<FuncOp> stmts;
    getAllScopStmts(callee, stmts, m);

    if (stmts.size() >= 2)
      pes.insert(callee);
  });
}

/// Assuming the memrefs at the top-level are not aliases.
/// Also assuming each scop.stmt will have its accessed memrefs once in its
/// interface.
static bool areScopStmtsSeparable(FuncOp f) {
  SetVector<Value> visited; // memrefs visited.
  SetVector<Value> conflicted;
  f.walk([&](mlir::CallOp caller) {
    if (!caller->hasAttr("scop.stmt"))
      return;

    for (Value arg : caller.getArgOperands())
      if (arg.getType().isa<MemRefType>()) {
        if (visited.count(arg))
          conflicted.insert(arg);
        visited.insert(arg);
      }
  });

  if (conflicted.empty())
    return true;

  LLVM_DEBUG({
    llvm::errs() << "\nConflicted memrefs:\n\n";
    for (Value memref : conflicted)
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

// We possibly won't need this complicated analysis.
#if 0
  // Get all the callers for the target stmt.
  SmallVector<mlir::CallOp> callers;
  f.walk([&](Operation *op) {
    mlir::CallOp caller = dyn_cast<mlir::CallOp>(op);
    if (!caller)
      return;
    if (caller.getCallee() == stmt.getName())
      callers.push_back(caller);
  });

  // Get the domain for each caller.
  FlatAffineConstraints domain;

  // Map the dim pos to caller argument index. This should be CONSISTENT
  // across multiple callers.
  MapVector<unsigned, unsigned> dimToArgIdx;

  // Target loop structure.
  SmallVector<mlir::AffineForOp> forOps;

  for (mlir::CallOp caller : callers) {
    LLVM_DEBUG({
      dbgs() << "Working on caller: ";
      caller.dump();
    });
    SmallVector<Operation *> forAndIfOps;
    getEnclosingAffineForAndIfOps(*caller.getOperation(), &forAndIfOps);

    if (forAndIfOps.empty()) {
      LLVM_DEBUG(llvm::errs()
                     << "Callers should be located within loops/ifs.\n";);
      return failure();
    }

    if (find_if(forAndIfOps, [](Operation *op) {
          return isa<mlir::AffineIfOp>(op);
        }) != forAndIfOps.end()) {
      LLVM_DEBUG(dbgs() << "We cannot deal with enclosing affine.if yet.\n");
      return failure();
    }

    FlatAffineConstraints cst;
    if (failed(getIndexSet(forAndIfOps, &cst))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Cannot get constraints from for-and-if ops.\n");
      return failure();
    }
    LLVM_DEBUG({
      dbgs() << "To merge with:";
      cst.dump();
    });

    // Here is the domain merging step -
    // When domain is empty, i.e., we're looking at the first caller instance to
    // the target stmt.
    if (domain.getNumConstraints() == 0) {
      domain = cst;

      // Build the dim pos to arg idx map.
      for (auto arg : enumerate(caller.getArgOperands())) {
        unsigned pos;
        if (domain.findId(arg.value(), &pos))
          dimToArgIdx[pos] = arg.index();
      }

      LLVM_DEBUG({
        dbgs() << "\nDim pos to caller arg idx:\n\n";
        for (auto &it : dimToArgIdx)
          dbgs() << "  " << it.first << " -> " << it.second << '\n';
      });

      // Put the for loop structure.
      for (Operation *op : forAndIfOps)
        forOps.push_back(cast<mlir::AffineForOp>(op));
    } else {
      // Check the dims being the same -
      if (cst.getNumDimIds() != domain.getNumDimIds()) {
        LLVM_DEBUG(dbgs() << "The constraints among callers have different "
                             "number of dims: "
                          << cst.getNumDimIds() << " expected "
                          << domain.getNumDimIds() << ".\n");
        return failure();
      }

      // Check the consistency of the mapping -
      for (unsigned pos = 0; pos < cst.getNumDimIds(); ++pos)
        if (cst.getIdValue(pos) != caller.getOperand(dimToArgIdx[pos])) {
          LLVM_DEBUG(dbgs()
                     << "The dimToArgIdx map is inconsistent across callers. "
                        "Dim at pos "
                     << pos << " should be " << cst.getIdValue(pos) << ", got "
                     << caller.getOperand(dimToArgIdx[pos]) << "\n");
          return failure();
        }

      // Merge the current caller into the target.
      // When saying merge, we practically create a new AffineForOp at each
      // level if necessary.
      unsigned d = 0;
      while (d < forOps.size() &&
             cst.getIdValue(d) == forOps[d].getInductionVar())
        ++d;

      // We should clone the whole dst loop after forOps[d].
      if (d < forOps.size()) {
        Value dstInd = cst.getIdValue(d);
        assert(isForInductionVar(dstInd));

        mlir::AffineForOp forOp = forOps[d];
        b.setInsertionPointAfter(forOp);

        Operation *cloned =
            b.clone(*getForInductionVarOwner(dstInd).getOperation());
        forOps[d] = cast<mlir::AffineForOp>(cloned);

        // Clean up the cloned AffineForOp to remove other statements.
        // As well as finding the new caller.
        SetVector<Operation *> toErase, invalidOps;
        mlir::CallOp newCaller = nullptr;
        bool hasExtraCallers = false;
        cloned->walk([&](Operation *op) {
          if (mlir::CallOp clonedCaller = dyn_cast<mlir::CallOp>(op)) {
            if (clonedCaller.getCallee() != stmt.getName())
              toErase.insert(op);
            else {
              if (!newCaller)
                newCaller = clonedCaller;
              else
                hasExtraCallers = true;
            }
          } else if (!isa<mlir::AffineForOp, mlir::AffineYieldOp>(op))
            invalidOps.insert(op);
        });

        if (!invalidOps.empty()) {
          LLVM_DEBUG({
            dbgs() << "There are invalid ops appeared in the cloned forOp:\n";
            for (Operation *op : invalidOps)
              op->dump();
          });
          return failure();
        }

        if (hasExtraCallers) {
          LLVM_DEBUG({
            dbgs()
                << "There are extra callers to the same stmt in the cloned:\n";
            cloned->dump();
          });
          return failure();
        }

        for (Operation *op : toErase)
          op->erase();

        // Since all the callers share the same loop depth, we don't need to
        // worry about this update.
        /// TODO: what if they don't?
        for (unsigned i = d + 1; i < forOps.size(); ++i)
          forOps[i] = cast<mlir::AffineForOp>(forAndIfOps[i]);

        // Now remove the old caller.
      }
    }
  }
#endif
}

/// The input function will be altered in-place.
static LogicalResult distributeScopStmts(
    FuncOp f, SmallVectorImpl<std::pair<FuncOp, SmallVector<unsigned>>> &dist,
    ModuleOp m, OpBuilder &b) {
  SetVector<FuncOp> stmts;
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
    SetVector<FuncOp> pes;
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
    SetVector<FuncOp> pesToProc;
    for (FuncOp pe : pes) {
      if (!areScopStmtsSeparable(pe)) {
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
  SetVector<FuncOp> stmts;
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
  SetVector<Operation *> erased;
  for (mlir::CallOp caller : callers) {
    SmallVector<Operation *> ops;
    getEnclosingAffineForAndIfOps(*caller.getOperation(), &ops);

    FlatAffineConstraints cst;
    getIndexSet(ops, &cst);

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
  SmallVector<FlatAffineConstraints, 4> csts;
  transform(innermosts, std::back_inserter(csts), [&](mlir::AffineForOp forOp) {
    FlatAffineConstraints cst;
    cst.addInductionVarOrTerminalSymbol(forOp.getInductionVar());
    // cst.addAffineForOpDomain(forOp);

    LLVM_DEBUG(cst.dump());

    return cst;
  });

  // Make every constraint has the same induction variable.
  for (unsigned i = 1; i < csts.size(); ++i)
    csts[i].setIdValue(0, csts[0].getIdValue(0));

  // Check if two loops have intersection.
  for (unsigned i = 0; i < csts.size(); ++i)
    for (unsigned j = i + 1; j < csts.size(); ++j) {
      FlatAffineConstraints tmp{csts[i]};
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
                SmallVector<AffineExpr> results;
                copy_if(ubMap.getResults(), std::back_inserter(results),
                        [&](AffineExpr expr) { return expr != ub; });
                AffineMap newUbMap =
                    AffineMap::get(ubMap.getNumDims(), ubMap.getNumSymbols(),
                                   results, ubMap.getContext());
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

void phism::registerLoopTransformPasses() {
  PassRegistration<AnnotatePointLoopsPass>(
      "annotate-point-loops", "Annotate loops with point/tile info.");
  PassRegistration<ExtractPointLoopsPass>(
      "extract-point-loops", "Extract point loop bands into functions");
  PassRegistration<RedistributeScopStatementsPass>(
      "redis-scop-stmts",
      "Redistribute scop statements across extracted point loops.");
  PassRegistration<LoopMergePass>("loop-merge",
                                  "Merge loops by affine analysis.");

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
        pm.addPass(std::make_unique<RedistributeScopStatementsPass>());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(std::make_unique<LoopMergePass>());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createInlinerPass());
      });
}
