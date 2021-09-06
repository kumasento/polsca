//===- ArrayPartitions.cc - Partitioning arrays ------------------ C++-===//

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

#define DEBUG_TYPE "loop-extract"

using namespace mlir;
using namespace llvm;
using namespace phism;

/// -------------------------- Dependence analysis ---------------------------

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

struct MemRefAccessInfo {
  CallOp caller;
  FlatAffineConstraints readCst;
  FlatAffineConstraints writeCst;

  MemRefAccessInfo(CallOp caller, FlatAffineConstraints readCst,
                   FlatAffineConstraints writeCst)
      : caller(caller), readCst(readCst), writeCst(writeCst) {}

  bool isReadOnly() const {
    return writeCst.getNumConstraints() == 0 && readCst.getNumConstraints() > 0;
  }

  bool isWriteOnly() const {
    return writeCst.getNumConstraints() > 0 && readCst.getNumConstraints() == 0;
  }

  bool isEmpty() const {
    return writeCst.getNumConstraints() == 0 &&
           readCst.getNumConstraints() == 0;
  }

  bool isReadWrite() const {
    return writeCst.getNumConstraints() > 0 && readCst.getNumConstraints() > 0;
  }

  unsigned getNumDims() const {
    return isReadOnly() ? readCst.getNumDimIds() : writeCst.getNumDimIds();
  }
};

static FlatAffineConstraints
getArrayPartition(const FlatAffineConstraints &cst, ArrayRef<int64_t> inds,
                  FuncOp callee, ArrayRef<Value> ivs,
                  SmallDenseMap<Value, unsigned> &ivIds) {
  FlatAffineConstraints cur{cst};
  for (auto ind : enumerate(inds)) {
    Value id = callee.getArgument(ivIds[ivs[ind.index()]]);
    unsigned pos;
    cur.findId(id, &pos);
    cur.setAndEliminate(pos, {ind.value()});
  }
  cur.removeRedundantConstraints();

  return cur;
}

/// TODO: this function tries to find whether the FORMs of two constraints are
/// the same, not exactly the domain they are covering.
static bool isSame(const FlatAffineConstraints &cst1,
                   const FlatAffineConstraints &cst2) {
  if (cst1.getNumCols() != cst2.getNumCols())
    return false;
  if (cst1.getNumConstraints() != cst2.getNumConstraints())
    return false;
  if (cst1.getNumEqualities() != cst2.getNumEqualities())
    return false;
  for (unsigned i = 0; i < (unsigned)cst1.getNumEqualities(); ++i)
    for (unsigned j = 0; j < (unsigned)cst2.getNumCols(); ++j)
      if (cst1.atEq(i, j) != cst2.atEq(i, j))
        return false;
  for (unsigned i = 0; i < (unsigned)cst1.getNumInequalities(); ++i)
    for (unsigned j = 0; j < (unsigned)cst2.getNumCols(); ++j)
      if (cst1.atIneq(i, j) != cst2.atIneq(i, j))
        return false;

  return true;
}

/// Map from memrefs to their accesses from all PE callers.
static auto getMemRefAccessInfo(ArrayRef<Operation *> callers, ModuleOp m) {
  SmallDenseMap<Value, std::vector<MemRefAccessInfo>> accesses;

  // Initialise the map from memref to PE caller accesses.
  for (Operation *op : callers) {
    mlir::CallOp caller = cast<mlir::CallOp>(op);
    FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));

    SmallVector<std::pair<Value, unsigned>> memrefs;
    for (auto arg : enumerate(caller.getArgOperands()))
      if (arg.value().getType().isa<MemRefType>())
        memrefs.push_back({arg.value(), arg.index()});

    // Iterate every memref being accessed by the current caller.
    for (auto memref : memrefs) {
      Value arg = callee.getArgument(memref.second);

      // Get all the read/write accesses.
      SmallVector<Operation *> loadOps, storeOps;
      copy_if(arg.getUsers(), std::back_inserter(loadOps),
              [](Operation *op) { return isa<mlir::AffineLoadOp>(op); });
      copy_if(arg.getUsers(), std::back_inserter(storeOps),
              [](Operation *op) { return isa<mlir::AffineStoreOp>(op); });

      // Union all the read constraints from all the load operations.
      FlatAffineConstraints readCst = getDomain(loadOps);
      FlatAffineConstraints writeCst = getDomain(storeOps);
      accesses[memref.first].push_back({caller, readCst, writeCst});
    }
  }

  return accesses;
}

static bool
getLoopIVsAndBounds(Operation *op, SmallVectorImpl<Value> &ivs,
                    SmallVectorImpl<std::pair<int64_t, int64_t>> &bounds) {
  SmallVector<AffineForOp> forOps;
  getLoopIVs(*op, &forOps);

  for (AffineForOp forOp : forOps)
    if (forOp.hasConstantUpperBound() && forOp.hasConstantLowerBound()) {
      bounds.push_back(
          {forOp.getConstantLowerBound(), forOp.getConstantUpperBound()});
      ivs.push_back(forOp.getInductionVar());
    }

  // If there is no bound, or not all bounds are constant, return false.
  if (bounds.empty() || bounds.size() != forOps.size())
    return false;

  return true;
}

static auto getArgIndexMap(CallOp caller) {
  SmallDenseMap<Value, unsigned> argId;
  for (auto arg : enumerate(caller.getArgOperands()))
    argId[arg.value()] = arg.index();
  return argId;
}

// Get the combinations of indices within the provided bounds.
static auto getIndexCombinations(ArrayRef<std::pair<int64_t, int64_t>> bounds) {
  std::vector<std::vector<int64_t>> indices{{}};
  for (auto bound : bounds) {
    int64_t lb, ub;
    std::tie(lb, ub) = bound;
    std::vector<std::vector<int64_t>> newIndices;

    for (auto index : indices)
      for (int64_t value = lb; value < ub; ++value) {
        std::vector<int64_t> newIndex{index};
        newIndex.push_back(value);
        newIndices.push_back(newIndex);
      }

    std::swap(indices, newIndices);
  }
  return indices;
}

using Partition = std::vector<std::pair<int64_t, int64_t>>;

static bool checkAccessOverlap(MemRefAccessInfo &info, ModuleOp m,
                               std::set<Partition> &partitions) {
  // Get the IVs and constant bounds for the loops surrounding the caller.
  SmallVector<Value> ivs;
  SmallVector<std::pair<int64_t, int64_t>> bounds;
  if (!getLoopIVsAndBounds(info.caller.getOperation(), ivs, bounds))
    return true;

  // Loop induction variable to argument ID in the caller.
  auto argId = getArgIndexMap(info.caller);

  // Get every partition.
  // TODO: Can we make this part less memory intensive?
  std::vector<std::vector<int64_t>> indices = getIndexCombinations(bounds);

  FlatAffineConstraints cst = info.isWriteOnly() ? info.writeCst : info.readCst;
  FuncOp callee = cast<FuncOp>(m.lookupSymbol(info.caller.getCallee()));

  // Iterate every possible partitions and check if they would overlap.
  std::vector<FlatAffineConstraints> parts; // temporary results;
  for (auto inds1 : indices) {
    FlatAffineConstraints cur1 =
        getArrayPartition(cst, inds1, callee, ivs, argId);
    if (cur1.isEmpty())
      continue;

    for (auto inds2 : indices) {
      if (inds1 == inds2)
        continue;

      FlatAffineConstraints cur2 =
          getArrayPartition(cst, inds2, callee, ivs, argId);

      // If cur1 and cur2 are exactly the same, then it shouldn't be
      // considered as overlapping.
      if (isSame(cur1, cur2))
        continue;

      FlatAffineConstraints tmp{cur1};
      tmp.append(cur2);
      if (!tmp.isEmpty())
        return true;
    }
    parts.push_back(cur1);
  }

  // De-duplicate
  // TODO: make it more efficient.
  for (unsigned i = 0; i < parts.size(); ++i) {
    std::vector<std::pair<int64_t, int64_t>> partition;

    for (unsigned pos = 0; pos < parts[i].getNumDimIds(); ++pos) {
      auto lb = parts[i].getConstantLowerBound(pos);
      auto ub = parts[i].getConstantUpperBound(pos);
      if (lb.hasValue() && ub.hasValue())
        partition.push_back({lb.getValue(), ub.getValue()});
    }

    if (partition.size() != parts[i].getNumDimIds()) {
      llvm::errs()
          << "The number of constant partitions are less than the dim Ids\n";
      return true;
    }

    partitions.insert(partition);
  }

  return false;
}

static void arrayPartition(FuncOp f, ModuleOp m, OpBuilder &b) {
  // Get all the PE callers.
  SmallVector<Operation *> callers;
  f.walk([&](CallOp caller) {
    if (caller->hasAttr("scop.pe"))
      callers.push_back(caller);
  });
  if (callers.empty())
    return;

  // Get MemRef accesses.
  auto accesses = getMemRefAccessInfo(callers, m);

  for (auto &access : accesses) {
    Value memref;
    std::vector<MemRefAccessInfo> accessInfos;
    std::tie(memref, accessInfos) = access;

    // For now, we only look at those memrefs that are only accessed by one PE.
    if (accessInfos.size() > 1)
      continue;

    // Check if the only access is read or write. We cannot deal with read-write
    // access at the moment.
    // TODO: deal with read-write.
    MemRefAccessInfo info = accessInfos.front();
    if (info.isEmpty() || info.isReadWrite())
      continue;

    // Check if there any overlap for the memref being accessed. If not, then we
    // can find valid array partitions from it.
    std::set<Partition> partitions;
    if (checkAccessOverlap(info, m, partitions))
      continue;

    memref.dump();
    llvm::errs() << "Partitions: \n";
    for (auto it : partitions) {
      for (auto bound : enumerate(it)) {
        llvm::errs() << bound.index() << " -> " << bound.value().first << ' '
                     << bound.value().second << "\n";
      }
      llvm::errs() << "-----------------\n";
    }

    // Next, decide how to update the type of the memref.
    // First determine what the partition size is for each dimension, then
    // decide new memref type.
    SmallVector<int64_t> dimSizes(info.getNumDims(), 0);
    for (unsigned i = 0; i < info.getNumDims(); ++i)
      for (auto p : partitions)
        dimSizes[i] = std::max((int64_t)dimSizes[i],
                               (int64_t)(p[i].second - p[i].first + 1));
  }
}

namespace {
struct ArrayPartitionPass
    : public mlir::PassWrapper<ArrayPartitionPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    m.walk([&](FuncOp f) { arrayPartition(f, m, b); });
  }
};

} // namespace

void phism::registerArrayPartitionPasses() {
  PassRegistration<ArrayPartitionPass>("array-partition", "Partition arrays");
}
