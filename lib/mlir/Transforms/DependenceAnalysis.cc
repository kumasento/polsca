//===- DependenceAnalysis.cc - Dependence analysis ----------------- C++-===//

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
#include <set>

#define DEBUG_TYPE "dependence-analysis"

using namespace mlir;
using namespace llvm;
using namespace phism;

/// -------------------------- Dependence analysis ---------------------------

static bool hasPeCaller(FuncOp f) {
  bool ret = false;
  f.walk([&](CallOp caller) {
    if (caller->hasAttr("scop.pe"))
      ret = true;
  });
  return ret;
}

static FuncOp getTopFunction(ModuleOp m) {
  FuncOp top = nullptr;
  m.walk([&](FuncOp f) {
    if (hasPeCaller(f)) {
      assert(!top && "There should be only one top function.");
      top = f;
    }
  });
  return top;
}

static FlatAffineConstraints getOpIndexSet(Operation *op) {
  FlatAffineConstraints cst;
  SmallVector<Operation *, 4> ops;
  getEnclosingAffineForAndIfOps(*op, &ops);
  getIndexSet(ops, &cst);
  return cst;
}

/// Get the access domain from all the provided operations.
static FlatAffineConstraints getDomain(ArrayRef<Operation *> ops) {
  FlatAffineConstraints cst;
  for (Operation *op : ops) {
    MemRefAccess access(op);

    AffineValueMap accessMap;
    access.getAccessMap(&accessMap);

    SmallSetVector<Value, 4> ivs; // used to access memref.
    for (Value operand : accessMap.getOperands())
      ivs.insert(operand);

    FlatAffineConstraints domain = getOpIndexSet(op);

    // project out those IDs that are not in the accessMap.
    SmallVector<Value> values;
    domain.getIdValues(0, domain.getNumDimIds(), &values);

    SmallVector<Value> toProject;
    for (Value value : values)
      if (!ivs.count(value))
        toProject.push_back(value);

    for (Value id : toProject) {
      unsigned pos;
      domain.findId(id, &pos);
      domain.projectOut(pos);
    }

    cst.mergeAndAlignIdsWithOther(0, &domain);
    cst.append(domain);
  }

  cst.removeTrivialRedundancy();
  return cst;
}

namespace {

/// Memory access.
struct PeMemAccess {
  Value memref;
  CallOp caller;
  FlatAffineConstraints read;
  FlatAffineConstraints write;

  PeMemAccess() {}
  PeMemAccess(Value memref, CallOp caller, FlatAffineConstraints read,
              FlatAffineConstraints write)
      : memref(memref), caller(caller), read(read), write(write) {}

  // Predicates.
  bool hasWrite() const { return write.getNumConstraints() > 0; }
  bool hasRead() const { return read.getNumConstraints() > 0; }
  bool noWrite() const { return write.getNumConstraints() == 0; }
  bool noRead() const { return read.getNumConstraints() == 0; }
  bool isReadOnly() const { return noWrite() && hasRead(); }
  bool isWriteOnly() const { return hasWrite() && noRead(); }
  bool isEmpty() const { return noWrite() && noRead(); }
  bool isReadWrite() const { return hasWrite() && hasRead(); }

  unsigned getNumDims() const {
    return isReadOnly() ? read.getNumDimIds() : write.getNumDimIds();
  }
};
} // namespace

static PeMemAccess getPeMemAccess(mlir::CallOp caller, unsigned argId,
                                  ModuleOp m) {
  FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));

  Value arg = callee.getArgument(argId);
  // Get all the read/write accesses.
  SmallVector<Operation *> loadOps, storeOps;
  copy_if(arg.getUsers(), std::back_inserter(loadOps),
          [](Operation *op) { return isa<mlir::AffineLoadOp>(op); });
  copy_if(arg.getUsers(), std::back_inserter(storeOps),
          [](Operation *op) { return isa<mlir::AffineStoreOp>(op); });

  // Union all the read constraints from all the load operations.
  FlatAffineConstraints read = getDomain(loadOps);
  FlatAffineConstraints write = getDomain(storeOps);

  Value memref = caller.getOperand(argId + 1);
  return {memref, caller, read, write};
}

namespace {

/// A single PE instance.
struct PeInst {
  CallOp caller;
  SmallVector<mlir::AffineForOp> forOps;
  SmallDenseMap<Value, PeMemAccess> memAccesses;

  PeInst() {}
  PeInst(CallOp caller) : caller(caller) { getLoopIVs(*caller, &forOps); }

  void initMemAccesses(ModuleOp m) {
    for (auto arg : enumerate(caller.getArgOperands()))
      if (arg.value().getType().isa<MemRefType>())
        memAccesses[arg.value()] = getPeMemAccess(caller, arg.index(), m);
  }
};

} // namespace

/// Get the list of memrefs being accessed by both the srcInst and the dstInst.
/// One of the access should be write.
static SmallVector<Value> getCommonMemRefs(PeInst &srcInst, PeInst &dstInst) {
  SmallVector<Value> memrefs;
  for (auto &it : srcInst.memAccesses)
    // if it.first (memref) exists in both the src and the dst.
    // The access should be at RAW, WAR, or WAW.
    if (dstInst.memAccesses.count(it.first) &&
        (dstInst.memAccesses[it.first].hasWrite() || it.second.hasWrite()))
      memrefs.push_back(it.first);
  return memrefs;
}

namespace {
struct PeMemDependence {
  enum Type { RAW, WAR, WAW };

  Type type;
  unsigned depth;
  FlatAffineConstraints src, dst;
  Operation *srcOp, *dstOp;

  PeMemDependence(Type type, unsigned depth, FlatAffineConstraints src,
                  FlatAffineConstraints dst, Operation *srcOp, Operation *dstOp)
      : type(type), depth(depth), src(src), dst(dst), srcOp(srcOp),
        dstOp(dstOp) {}
};
} // namespace

// Returns the number of outer loop common to 'src/dstDomain'.
// Loops common to 'src/dst' domains are added to 'commonLoops' if non-null.
static unsigned getNumCommonLoops(const FlatAffineConstraints &srcDomain,
                                  const FlatAffineConstraints &dstDomain) {
  // Find the number of common loops shared by src and dst accesses.
  unsigned minNumLoops =
      std::min(srcDomain.getNumDimIds(), dstDomain.getNumDimIds());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (!isForInductionVar(srcDomain.getIdValue(i)) ||
        !isForInductionVar(dstDomain.getIdValue(i)) ||
        srcDomain.getIdValue(i) != dstDomain.getIdValue(i))
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

static Block *getCommonBlock(Operation *src, Operation *dst,
                             const FlatAffineConstraints &srcDomain,
                             unsigned numCommonLoops) {
  // Get the chain of ancestor blocks to the given `MemRefAccess` instance. The
  // search terminates when either an op with the `AffineScope` trait or
  // `endBlock` is reached.
  auto getChainOfAncestorBlocks = [&](Operation *op,
                                      SmallVector<Block *, 4> &ancestorBlocks,
                                      Block *endBlock = nullptr) {
    Block *currBlock = op->getBlock();
    // Loop terminates when the currBlock is nullptr or equals to the endBlock,
    // or its parent operation holds an affine scope.
    while (currBlock && currBlock != endBlock &&
           !currBlock->getParentOp()->hasTrait<OpTrait::AffineScope>()) {
      ancestorBlocks.push_back(currBlock);
      currBlock = currBlock->getParentOp()->getBlock();
    }
  };

  if (numCommonLoops == 0) {
    Block *block = src->getBlock();
    while (!llvm::isa<FuncOp>(block->getParentOp())) {
      block = block->getParentOp()->getBlock();
    }
    return block;
  }
  Value commonForIV = srcDomain.getIdValue(numCommonLoops - 1);
  AffineForOp forOp = getForInductionVarOwner(commonForIV);
  assert(forOp && "commonForValue was not an induction variable");

  // Find the closest common block including those in AffineIf.
  SmallVector<Block *, 4> srcAncestorBlocks, dstAncestorBlocks;
  getChainOfAncestorBlocks(src, srcAncestorBlocks, forOp.getBody());
  getChainOfAncestorBlocks(dst, dstAncestorBlocks, forOp.getBody());

  Block *commonBlock = forOp.getBody();
  for (int i = srcAncestorBlocks.size() - 1, j = dstAncestorBlocks.size() - 1;
       i >= 0 && j >= 0 && srcAncestorBlocks[i] == dstAncestorBlocks[j];
       i--, j--)
    commonBlock = srcAncestorBlocks[i];

  return commonBlock;
}

static bool
srcAppearsBeforeDstInAncestralBlock(Operation *src, Operation *dst,
                                    const FlatAffineConstraints &srcDomain,
                                    unsigned numCommonLoops) {
  // Get Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
  auto *commonBlock = getCommonBlock(src, dst, srcDomain, numCommonLoops);
  // Check the dominance relationship between the respective ancestors of the
  // src and dst in the Block of the innermost among the common loops.
  auto *srcInst = commonBlock->findAncestorOpInBlock(*src);
  assert(srcInst != nullptr);
  auto *dstInst = commonBlock->findAncestorOpInBlock(*dst);
  assert(dstInst != nullptr);

  // Determine whether dstInst comes after srcInst.
  return srcInst->isBeforeInBlock(dstInst);
}

/// TODO: skip some of the dependencies if they are not valid.
/// We need to make sure srcOp should be the ancestor of dstOp,
static void addOrSkipPeMemDependence(PeMemDependence::Type type, unsigned depth,
                                     FlatAffineConstraints src,
                                     FlatAffineConstraints dst,
                                     Operation *srcOp, Operation *dstOp,
                                     SmallVectorImpl<PeMemDependence> &deps) {
  FlatAffineConstraints srcDomain = getOpIndexSet(srcOp),
                        dstDomain = getOpIndexSet(dstOp);

  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  // If we are looking at the dependence at a deeper level, we should make sure
  // srcOp properly dominates dstOp.
  if (numCommonLoops < depth && !srcAppearsBeforeDstInAncestralBlock(
                                    srcOp, dstOp, srcDomain, numCommonLoops))
    return;

  deps.push_back({type, depth, src, dst, srcOp, dstOp});
}

static auto getPeMemDependenceMap(ArrayRef<Value> commonMemRefs, unsigned depth,
                                  PeInst &srcInst, PeInst &dstInst) {
  SmallDenseMap<Value, SmallVector<PeMemDependence, 4>> deps;
  for (Value memref : commonMemRefs) {
    if (srcInst.memAccesses[memref].hasWrite() &&
        dstInst.memAccesses[memref].hasRead())
      addOrSkipPeMemDependence(PeMemDependence::Type::RAW, depth,
                               srcInst.memAccesses[memref].write,
                               dstInst.memAccesses[memref].read, srcInst.caller,
                               dstInst.caller, deps[memref]);
    if (srcInst.memAccesses[memref].hasRead() &&
        dstInst.memAccesses[memref].hasWrite())
      addOrSkipPeMemDependence(PeMemDependence::Type::WAR, depth,
                               srcInst.memAccesses[memref].read,
                               dstInst.memAccesses[memref].write,
                               srcInst.caller, dstInst.caller, deps[memref]);
    if (srcInst.memAccesses[memref].hasWrite() &&
        dstInst.memAccesses[memref].hasWrite())
      addOrSkipPeMemDependence(PeMemDependence::Type::WAW, depth,
                               srcInst.memAccesses[memref].write,
                               dstInst.memAccesses[memref].write,
                               srcInst.caller, dstInst.caller, deps[memref]);

    LLVM_DEBUG({
      memref.dump();
      for (auto &dep : deps[memref])
        llvm::errs() << "Dependence type: " << dep.type << '\n';
    });

    if (deps[memref].empty())
      deps.erase(deps.find(memref));
  }

  return deps;
}

static DependenceResult checkPeInstDependence(PeInst &srcInst, PeInst &dstInst,
                                              unsigned depth) {
  LLVM_DEBUG(llvm::errs() << "=========\nDependence analysis:\n";);
  LLVM_DEBUG(srcInst.caller.dump(););
  LLVM_DEBUG(dstInst.caller.dump(););

  SmallVector<Value> commonMemRefs = getCommonMemRefs(srcInst, dstInst);

  LLVM_DEBUG({
    llvm::errs() << "Common memrefs:\n";
    for (Value memref : commonMemRefs)
      memref.dump();
    llvm::errs() << "--------------------------\n";
  });

  // If there is no common memrefs, no dependence.
  if (commonMemRefs.empty())
    return DependenceResult::NoDependence;

  // Get all kinds of dependencies for each memref between the src and dst.
  auto deps = getPeMemDependenceMap(commonMemRefs, depth, srcInst, dstInst);
  if (deps.empty()) {
    LLVM_DEBUG(llvm::errs() << "No dependences for every common memref.\n";);
    return DependenceResult::NoDependence;
  }

  // We know that the loops wrapping srcInst and dstInst have constant bounds.
  // So they can be fully unrolled. For each dependence in the map, we can
  // realise them by the actual outer loop induction variable values.

  return DependenceResult::NoDependence;
}

namespace {

struct DependenceAnalysisPass
    : public mlir::PassWrapper<DependenceAnalysisPass,
                               OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    FuncOp top = getTopFunction(m);

    SmallVector<PeInst, 4> peInsts;
    top.walk([&](mlir::CallOp caller) {
      if (caller->hasAttr("scop.pe"))
        peInsts.push_back({caller});
    });

    // initialize memory accesses.
    for (PeInst &inst : peInsts)
      inst.initMemAccesses(m);

    // calculate dependencies
    unsigned maxLoopDepth = 3;
    for (unsigned depth = 1; depth <= maxLoopDepth; ++depth)
      for (PeInst &srcInst : peInsts)
        for (PeInst &dstInst : peInsts)
          checkPeInstDependence(srcInst, dstInst, depth);

    // Debugging.
    llvm::errs() << "PE callers:\n";
    for (PeInst &inst : peInsts) {
      inst.caller.dump();

      llvm::errs() << "Memory accesses:\n";
      for (auto &it : inst.memAccesses) {
        llvm::errs() << "Mem:\n";
        it.first.dump();

        llvm::errs() << "Read:\n";
        it.second.read.dump();
        llvm::errs() << "Write:\n";
        it.second.write.dump();

        llvm::errs() << "-------------------------\n";
      }
    }
  }
};

} // namespace

void phism::registerDependenceAnalysisPasses() {
  PassRegistration<DependenceAnalysisPass>("dependence-analysis",
                                           "Analyse PE dependencies");
}
