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

#define DEBUG_TYPE "array-partition"

using namespace mlir;
using namespace llvm;
using namespace phism;

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

/// ---------------- Simple array partition ---------------------------
/// Find partitioning opportunities by affine expressions.

/// If the affine map is not single constant, we filter out the rest of the
/// constants.
static AffineMap filterExtraConstantResults(AffineMap affMap) {
  if (affMap.isSingleConstant())
    return affMap;

  SmallVector<AffineExpr> results;
  for (AffineExpr result : affMap.getResults()) {
    if (result.isa<AffineConstantExpr>())
      continue;
    results.push_back(result);
  }

  return AffineMap::get(affMap.getNumDims(), affMap.getNumSymbols(), results,
                        affMap.getContext());
}

static bool isTiledLoopBound(AffineMap affMap) {
  if (affMap.getNumResults() != 1)
    return false;

  AffineExpr expr = affMap.getResult(0);
  if (!expr.isSymbolicOrConstant())
    return false;
  if (affMap.getNumSymbols() != 1)
    return false;

  SmallVector<int64_t> flattened;
  getFlattenedAffineExpr(expr, 0, 1, &flattened);

  // flattened = {tile size, 0 or tile size}
  if (flattened.size() != 2 || flattened[0] <= 0 || flattened[1] < 0 ||
      (flattened[1] > 0 && flattened[1] != flattened[0]))
    return false;

  return true;
}

static int64_t getTileSize(AffineMap affMap) {
  assert(isTiledLoopBound(affMap));
  AffineExpr expr = affMap.getResult(0);

  SmallVector<int64_t> flattened;
  getFlattenedAffineExpr(expr, 0, 1, &flattened);

  return flattened[0];
}

namespace {
struct TileInfo {
  SmallVector<int64_t> sizes;
  Value memref;

  TileInfo() {}
  TileInfo(SmallVector<int64_t> tileSizes, Value memref)
      : sizes{tileSizes.begin(), tileSizes.end()}, memref{memref} {}
};
} // namespace

/// Tile the input MemRefType statically.
static MemRefType getTiledMemRefType(const TileInfo &tileInfo,
                                     bool tileOnly = false) {
  MemRefType src = tileInfo.memref.getType().cast<MemRefType>();
  assert(src.getAffineMaps().empty() &&
         "We don't support memref with affine maps.");

  SmallVector<int64_t> dstShape;
  auto srcShape = src.getShape();
  if (!tileOnly)
    for (unsigned i = 0; i < srcShape.size(); ++i)
      dstShape.push_back(
          (int64_t)ceil((double)srcShape[i] / tileInfo.sizes[i]));
  for (unsigned i = 0; i < srcShape.size(); ++i)
    dstShape.push_back(tileInfo.sizes[i]);

  return MemRefType::get(dstShape, src.getElementType());
}

namespace {

static MapVector<Value, TileInfo> getTilingInfo(ArrayRef<Value> memrefs,
                                                ModuleOp m) {
  MapVector<Value, TileInfo> tiling;
  // See if they have simple access patterns that can be directly extracted.
  for (Value memref : memrefs) {
    // Check if all the users of memref are scop.pe callers.
    if (any_of(memref.getUsers(), [&](Operation *op) {
          return !isa<CallOp>(op) || !op->hasAttr("scop.pe");
        })) {
      LLVM_DEBUG({
        memref.dump();
        llvm::errs() << " has been skipped since it has non PE caller users.\n";
      });
      continue;
    }

    // Get all the memory accesses.
    SmallVector<MemRefAccess> accesses;
    for (Operation *op : memref.getUsers()) {
      CallOp caller = cast<CallOp>(op);
      FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));

      unsigned argId =
          find(caller.getArgOperands(), memref) - caller.arg_operand_begin();
      // memref counterpart in the callee.
      Value arg = callee.getArgument(argId);
      assert(arg.getType() == memref.getType() &&
             "The type of the caller/callee arg should match.");

      callee.walk([&](AffineLoadOp loadOp) {
        if (loadOp.getMemRef() == arg)
          accesses.push_back(MemRefAccess(loadOp));
      });
      callee.walk([&](AffineStoreOp storeOp) {
        if (storeOp.getMemRef() == arg)
          accesses.push_back(MemRefAccess(storeOp));
      });
    }

    // Debug the accesses.
    LLVM_DEBUG({
      memref.dump();
      for (MemRefAccess &access : accesses) {
        access.opInst->dump();
      }
    });

    // Check if all accesses are idenity maps.
    if (any_of(accesses, [&](MemRefAccess &access) {
          AffineValueMap vmap;
          access.getAccessMap(&vmap);
          return !vmap.getAffineMap().isIdentity();
        })) {
      LLVM_DEBUG(
          llvm::errs()
              << "Discontinued since there are non-identity access maps.\n";);
      continue;
    }

    // Check whether the affine bounds of the loop induction variables are the
    // same among all.
    // For each dimension of the memref, get the lbMap/ubMap from the
    // corresponding for-loop.

    // If for every access the maps at each dim is the same.
    bool isIdentical = true;
    SmallVector<AffineMap> lbMaps, ubMaps;
    for (MemRefAccess &ma : accesses) {
      AffineValueMap avm;
      ma.getAccessMap(&avm);

      SmallVector<AffineForOp> forOps;
      for (Value operand : avm.getOperands())
        if (isForInductionVar(operand))
          forOps.push_back(getForInductionVarOwner(operand));

      SmallVector<AffineMap> tmpLbMaps, tmpUbMaps;
      for (AffineForOp forOp : forOps) {
        // Filter out the result that are constants. We don't care about them.
        // ()[s0] -> (70, s0 * 32 + 32) will be ()[s0] -> (s0 * 32 + 32)
        tmpLbMaps.push_back(
            filterExtraConstantResults(forOp.getLowerBoundMap()));
        tmpUbMaps.push_back(
            filterExtraConstantResults(forOp.getUpperBoundMap()));
      }

      // Simply ignore those with constant lower upper bounds.
      // They won't cause much trouble (heuristically) if we don't partition
      // for them.
      if (any_of(tmpLbMaps,
                 [&](AffineMap am) { return am.isSingleConstant(); }) ||
          any_of(tmpUbMaps,
                 [&](AffineMap am) { return am.isSingleConstant(); })) {
        LLVM_DEBUG({
          llvm::errs() << "Skipped the access due to constant bounds: ";
          ma.opInst->dump();
        });
        continue;
      }

      if (lbMaps.empty()) {
        std::swap(tmpLbMaps, lbMaps);
        std::swap(tmpUbMaps, ubMaps);
      } else {
        isIdentical = tmpLbMaps == lbMaps && tmpUbMaps == ubMaps;
        if (!isIdentical)
          break;
      }
    }

    // Invalid results.
    assert(lbMaps.size() == ubMaps.size());
    if (lbMaps.empty()) // If it is all constant, we don't need to partition.
      continue;
    // There might be conflicting affine maps. Won't proceed.
    if (!isIdentical) {
      LLVM_DEBUG(llvm::errs()
                     << "Affine accesses are not identical in loop bounds.\n";);
      continue;
    }
    // Check if every bound has singular result.
    if (any_of(lbMaps, [&](AffineMap am) { return am.getNumResults() > 1; }) ||
        any_of(ubMaps, [&](AffineMap am) { return am.getNumResults() > 1; })) {
      LLVM_DEBUG(llvm::errs() << "There are loop bounds have more than one "
                                 "non-constant expressions.\n";);
      continue;
    }
    // Check whether the loop bounds satisfy the tiling constraints.
    if (any_of(lbMaps, [&](AffineMap am) { return !isTiledLoopBound(am); }) ||
        any_of(ubMaps, [&](AffineMap am) { return !isTiledLoopBound(am); })) {
      LLVM_DEBUG(llvm::errs() << "Loop bounds are not tiled expressions.\n";);
      continue;
    }

    // Finally resolve the tile size.
    SmallVector<int64_t> tileSizes;
    for (unsigned i = 0; i < lbMaps.size(); ++i) {
      int64_t tileSize = getTileSize(lbMaps[i]);
      if (tileSize != getTileSize(ubMaps[i]))
        continue;
      tileSizes.push_back(tileSize);
    }

    // Abandon further processing if the tile size cannot match memref's type.
    if ((int64_t)tileSizes.size() !=
        memref.getType().cast<MemRefType>().getRank()) {
      llvm::errs() << "Tile sizes are not equal to the rank of the memref.\n";
      continue;
    }

    // The resolved memref tiling.
    tiling[memref] = TileInfo{tileSizes, memref};
  }

  return tiling;
}

static FuncOp tileTopFunction(FuncOp top, ArrayRef<Value> memrefs,
                              MapVector<Value, TileInfo> &tiling, ModuleOp m,
                              OpBuilder &b) {
  // Things to process.
  SmallVector<Value> worklist;
  for (auto &it : tiling)
    worklist.push_back(it.first);

  // Next, we will resolve the memory tiling one by one -
  FuncOp prevFunc = top;

  for (unsigned stage = 0; stage < worklist.size(); ++stage) {
    LLVM_DEBUG({ llvm::errs() << ">>> Processing stage: " << stage << '\n'; });

    // Rebuild the caller list.
    SmallVector<mlir::CallOp> callers;
    prevFunc.walk([&](mlir::CallOp caller) {
      if (caller->hasAttr("scop.pe"))
        callers.push_back(caller);
    });

    Value memref = worklist[stage];
    TileInfo tileInfo = tiling[memref];

    LLVM_DEBUG({
      llvm::errs() << "Tiling memref: ";
      memref.dump();
      llvm::errs() << " into [ ";
      for (auto size : tileInfo.sizes)
        llvm::errs() << size << " ";
      llvm::errs() << "]\n";
    });

    // -------------------------------------------------------------------
    // Step 1: create a function of with an interface of the tiled input.
    MemRefType newMemRefType = getTiledMemRefType(tileInfo);
    LLVM_DEBUG({
      llvm::errs() << "New MemRef type: ";
      newMemRefType.dump();
      llvm::errs() << '\n';
    });

    // Function argument types. The old memref has been replaced.
    SmallVector<Type> newArgTypes;
    for (auto arg : prevFunc.getArguments()) {
      if (arg == memref)
        newArgTypes.push_back(newMemRefType);
      else
        newArgTypes.push_back(arg.getType());
    }

    // New function type.
    FunctionType newFuncType =
        b.getFunctionType(newArgTypes, prevFunc->getResultTypes());
    LLVM_DEBUG({
      llvm::errs() << "New function type: ";
      newFuncType.dump();
      llvm::errs() << '\n';
    });

    // Create the function with a __tiled suffix.
    b.setInsertionPointAfter(prevFunc.getOperation());
    FuncOp newFunc = b.create<FuncOp>(prevFunc.getLoc(),
                                      std::string(prevFunc.getName()) + "_" +
                                          std::to_string(stage),
                                      newFuncType);
    SmallVector<DictionaryAttr> argAttrs;
    prevFunc.getAllArgAttrs(argAttrs);
    newFunc.setAllArgAttrs(argAttrs);
    Block *entry = newFunc.addEntryBlock();
    b.setInsertionPointToEnd(entry);
    b.create<mlir::ReturnOp>(prevFunc.getLoc());

    // Map from the old callee to the new one.
    SmallDenseMap<FuncOp, FuncOp> calleeMap;

    // Create the __tiled version for each PE that has been affected by the
    // tiling, i.e., uses the memref.
    for (mlir::CallOp caller : callers) {
      // This caller should use the target memref.
      if (find(caller.getArgOperands(), memref) == caller.arg_operand_end())
        continue;

      FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));

      // Get the type of a MemRef tile.
      MemRefType newMemRefType =
          getTiledMemRefType(tileInfo, /*tileOnly=*/true);

      // New callee argument types.
      SmallVector<Type> newArgTypes;
      for (auto arg : caller.getArgOperands()) {
        if (arg == memref)
          newArgTypes.push_back(newMemRefType);
        else
          newArgTypes.push_back(arg.getType());
      }

      unsigned memId =
          find(caller.getArgOperands(), memref) - caller.arg_operand_begin();

      // New callee function type.
      FunctionType newFuncType =
          b.getFunctionType(newArgTypes, callee->getResultTypes());
      b.setInsertionPointAfter(callee);
      FuncOp newCallee = b.create<FuncOp>(callee.getLoc(),
                                          std::string(callee.getName()) + "_" +
                                              std::to_string(stage),
                                          newFuncType);
      Block *entry = newCallee.addEntryBlock();
      b.setInsertionPointToEnd(entry);
      b.create<mlir::ReturnOp>(callee.getLoc());

      // Argument map.
      BlockAndValueMapping vmap;
      vmap.map(callee.getArguments(), newCallee.getArguments());

      // Iterate every operation in the original callee and clone it to the
      // new one.
      b.setInsertionPointToStart(entry);
      for (Operation &op : callee.getBlocks().begin()->getOperations()) {
        if (isa<mlir::ReturnOp>(op))
          continue;
        b.clone(op, vmap);
      }

      // Rewrite the loop iterators for memory accesses.
      // For now I think the new iterator should be %i mod tile_size.
      // So we would simply create the corresponding new iterators, and use
      // them to replace the old ones applied to the tiled memref.
      newCallee.walk([&](Operation *op) {
        if (!isa<AffineLoadOp, AffineStoreOp>(op))
          return;
        // This affine.load/store op should have accessed the target memref.
        if (find(op->getOperands(), newCallee.getArgument(memId)) ==
            op->operand_end())
          return;

        b.setInsertionPoint(op);
        unsigned dim = 0; // the current memref dim.
        for (auto operand : enumerate(op->getOperands())) {
          if (!isForInductionVar(operand.value()))
            continue;

          // The affine map that does the modulo operation.
          AffineExpr modExpr = b.getAffineDimExpr(0) %
                               b.getAffineConstantExpr(tileInfo.sizes[dim]);
          AffineMap affMap = AffineMap::get(1, 0, modExpr);
          AffineApplyOp newInd =
              b.create<AffineApplyOp>(op->getLoc(), affMap, operand.value());

          op->setOperand(operand.index(), newInd);
          ++dim;
        }
      });

      // Finalise the result to the map.
      calleeMap.insert({callee, newCallee});
    }

    LLVM_DEBUG({
      for (auto &it : calleeMap) {
        llvm::errs() << it.first.getName() << " mapped to "
                     << it.second.getName() << "\n";
      }
    });

    // Argument map.
    BlockAndValueMapping vmap;
    vmap.map(prevFunc.getArguments(), newFunc.getArguments());

    // Iterate every operation in the original callee and clone it to the
    // new one.
    b.setInsertionPointToStart(entry);
    for (Operation &op : prevFunc.getBlocks().begin()->getOperations()) {
      if (isa<mlir::ReturnOp>(op))
        continue;
      b.clone(op, vmap);
    }

    SmallVector<mlir::CallOp> toRemove;

    // Replace the callers in the cloned function.
    newFunc.walk([&](mlir::CallOp caller) {
      // Try to see if the caller reaches to the tiled version.
      FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
      if (!calleeMap.count(callee)) // not affected by tiling
        return;

      // Now we replace the original caller to call the new callee, with the
      // memref argument replaced.
      FuncOp newCallee = calleeMap[callee];

      Value newMemRef = vmap.lookup(memref);

      SmallVector<Value> args;
      for (auto arg : enumerate(caller.getArgOperands())) {
        if (arg.value() != newMemRef)
          args.push_back(arg.value());
        else {
          // Create the caller that use a SUBVIEW of the original memref.
          b.setInsertionPoint(caller);
          SmallVector<OpFoldResult> offsets, sizes, strides;

          // Get the static rank.
          MemRefType newMemRefType = newMemRef.getType().cast<MemRefType>();
          unsigned rank = newMemRefType.getRank();

          // Look into the callee to find which tile loop has been used to
          // access the corresponding tiled memory dimensions.
          SmallVector<Value> indices;
          newCallee.walk([&](Operation *op) {
            if (!isa<mlir::AffineLoadOp, AffineStoreOp>(op))
              return;
            if (mlir::AffineLoadOp loadOp = dyn_cast<mlir::AffineLoadOp>(op))
              if (loadOp.getMemRef() != newCallee.getArgument(arg.index()))
                return;
            if (mlir::AffineStoreOp storeOp = dyn_cast<mlir::AffineStoreOp>(op))
              if (storeOp.getMemRef() != newCallee.getArgument(arg.index()))
                return;

            SmallVector<Value> tmpIndices;

            for (unsigned i = 1; i < op->getNumOperands(); ++i) {
              Value operand = op->getOperand(i);

              // The index for a tiled memref will be from an affine.apply op.
              AffineApplyOp applyOp = operand.getDefiningOp<AffineApplyOp>();
              if (!applyOp)
                continue;
              assert(applyOp.getNumOperands() == 1);

              Value indvar = applyOp.getOperand(0);

              mlir::AffineForOp forOp = getForInductionVarOwner(indvar);
              // forOp.dump();
              assert(forOp.getLowerBoundOperands().size() == 1 ||
                     forOp.getUpperBoundOperands().size() == 1);

              Value source = forOp.getUpperBoundOperands().size() == 1
                                 ? forOp.getUpperBoundOperands()[0]
                                 : forOp.getLowerBoundOperands()[0];
              assert(forOp.getLowerBoundOperands().size() < 1 ||
                     source == forOp.getLowerBoundOperands()[0]);

              tmpIndices.push_back(source);
            }

            if (indices.empty())
              std::swap(tmpIndices, indices);
            else {
              assert(tmpIndices == indices);
              std::swap(tmpIndices, indices);
            }
          });
          assert(indices.size() == rank / 2);

          // Figure out the offsets.
          // Need to know which tile loop has used by the accessed memref.
          for (unsigned i = 0; i < rank / 2; ++i) {
            unsigned pos = find(newCallee.getArguments(), indices[i]) -
                           newCallee.args_begin();
            assert(pos < newCallee.getNumArguments());
            offsets.push_back(caller.getOperand(pos));
          }
          for (unsigned i = 0; i < rank / 2; ++i)
            offsets.push_back(b.getIndexAttr(0));

          // Figure out sizes.
          for (unsigned i = 0; i < rank / 2; ++i)
            sizes.push_back(b.getIndexAttr(1));
          for (unsigned i = 0; i < rank / 2; ++i)
            sizes.push_back(
                b.getIndexAttr(newMemRefType.getShape()[i + rank / 2]));

          // Figure out strides.
          for (unsigned i = 0; i < rank; ++i)
            strides.push_back(b.getIndexAttr(1));

          // Get the resulting type for the subview. Otherwise, it won't
          // match.
          MemRefType newTiledMemRefType =
              memref::SubViewOp::inferRankReducedResultType(
                  rank / 2, newMemRef.getType().cast<MemRefType>(), offsets,
                  sizes, strides)
                  .cast<MemRefType>();

          // The final subview operaion.
          memref::SubViewOp subView =
              b.create<memref::SubViewOp>(caller.getLoc(), newTiledMemRefType,
                                          newMemRef, offsets, sizes, strides);
          subView.dump();

          // Strip the affine map
          MemRefType castMemRefType =
              newCallee.getArgument(arg.index()).getType().cast<MemRefType>();
          memref::CastOp cast = b.create<memref::CastOp>(
              caller.getLoc(), subView, castMemRefType);

          args.push_back(cast);
        }
      }

      // Note that we still use the original callee symbol here.
      mlir::CallOp newCaller =
          b.create<mlir::CallOp>(caller.getLoc(), newCallee, args);
      newCaller->setAttr("scop.pe", b.getUnitAttr());
      toRemove.push_back(caller);
    });

    // Clean up the callers.
    for (Operation *op : toRemove)
      op->erase();

    // Update the worklist to use the latest memref.
    for (unsigned j = stage + 1; j < worklist.size(); ++j) {
      tiling[vmap.lookup(worklist[j])] = tiling[worklist[j]];
      worklist[j] = vmap.lookup(worklist[j]);
    }

    newFunc.dump();

    prevFunc = newFunc;
  }

  return prevFunc;
}

/// Put all the visited functions into a set.
static auto markCalledFunctions(FuncOp top, ModuleOp m) {
  SmallPtrSet<FuncOp, 4> visited;
  visited.insert(top);

  top.walk([&](CallOp caller) {
    visited.insert(cast<FuncOp>(m.lookupSymbol(caller.getCallee())));
  });

  return visited;
}

/// Erase all the other functions.
static void sweepUncalledFunctions(ModuleOp m,
                                   const SmallPtrSetImpl<FuncOp> &visited) {
  SmallVector<FuncOp> toErase;
  m.walk([&](FuncOp f) {
    if (!visited.count(f))
      toErase.push_back(f);
  });

  for (FuncOp f : toErase)
    f.erase();
}

static void renameTiledFunctions(ModuleOp m, OpBuilder &b) {
  StringMap<std::string> newNames;

  m.walk([&](FuncOp f) {
    auto name = f.getName();

    SmallVector<StringRef> segments;
    llvm::SplitString(name, segments, "_");

    int i = (int)segments.size() - 1;
    for (; i >= 0; --i)
      if (segments[i].empty() ||
          !std::all_of(segments[i].begin(), segments[i].end(), ::isdigit))
        break;

    std::string newName;
    for (int j = 0; j <= i; ++j)
      newName += std::string(segments[j]) + "_";
    newName.pop_back();

    newNames.insert({name, newName});
    f.setName(newName);
  });

  // Make sure the callers call the correct function.
  m.walk([&](mlir::CallOp caller) {
    if (newNames.count(caller.getCallee()))
      caller->setAttr("callee",
                      b.getSymbolRefAttr(newNames[caller.getCallee()]));
  });
}

struct SimpleArrayPartitionPass
    : public PassWrapper<SimpleArrayPartitionPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    FuncOp top = getTopFunction(m);
    if (!top) {
      m.emitRemark() << "No top function found for array partition. Have you "
                        "forgot to annotate {scop.pe} to callers?\n";
      return;
    }

    SmallVector<CallOp> callers;
    top.walk([&](CallOp caller) {
      if (caller->hasAttr("scop.pe"))
        callers.push_back(caller);
    });

    if (callers.empty())
      return;

    // Get all the memrefs that can be partitioned.
    // TODO: consider scratchpad as well?
    SmallVector<Value> memrefs;
    for (Value arg : top.getArguments())
      if (arg.getType().isa<MemRefType>())
        memrefs.push_back(arg);

    // Get the tiling info.
    auto tiling = getTilingInfo(memrefs, m);

    // Tile the top function.
    FuncOp newTop = tileTopFunction(top, memrefs, tiling, m, b);

    // Clean up.
    sweepUncalledFunctions(m, markCalledFunctions(newTop, m));

    // Reset names.
    renameTiledFunctions(m, b);
  }
};
} // namespace

void phism::registerArrayPartitionPasses() {
  PassRegistration<ArrayPartitionPass>("array-partition", "Partition arrays");
  PassPipelineRegistration<>(
      "simple-array-partition", "Partition arrays", [&](OpPassManager &pm) {
        pm.addPass(std::make_unique<SimpleArrayPartitionPass>());
        pm.addPass(createCanonicalizerPass());
      });
}
