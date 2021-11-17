//===- SimpleArrayPartitions.cc - v0 array partition ----------------- C++-===//

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
#include "llvm/ADT/StringExtras.h"

#include <fstream>
#include <queue>
#include <set>

#define DEBUG_TYPE "array-partition"

using namespace mlir;
using namespace llvm;
using namespace phism;

namespace {
struct ArrayPartitionPipelineOptions
    : public mlir::PassPipelineOptions<ArrayPartitionPipelineOptions> {
  Option<bool> dumpFile{
      *this, "dumpFile",
      llvm::cl::desc("Enable dumping the tile info into a file."),
      llvm::cl::init(false)};
  Option<bool> flatten{*this, "flatten",
                       llvm::cl::desc("Enable flattening the partition dims."),
                       llvm::cl::init(false)};
};
} // namespace

static unsigned getPosition(AffineExpr expr) {
  if (auto dim = expr.dyn_cast<AffineDimExpr>())
    return dim.getPosition();
  if (auto symbol = expr.dyn_cast<AffineSymbolExpr>())
    return symbol.getPosition();
  return std::numeric_limits<unsigned>::max();
}

/// ---------------- Simple array partition ---------------------------
/// Find partitioning opportunities by affine expressions.

static bool isTiledLoopBound(AffineMap affMap) {
  if (affMap.getNumResults() != 1)
    return false;

  AffineExpr expr = affMap.getResult(0);
  if (!expr.isSymbolicOrConstant())
    return false;
  if (affMap.getNumSymbols() != 1)
    return false;

  SmallVector<int64_t> flattened;
  assert(succeeded(getFlattenedAffineExpr(expr, 0, 1, &flattened)));

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
  assert(succeeded(getFlattenedAffineExpr(expr, 0, 1, &flattened)));

  return flattened[0];
}

namespace {
struct TileInfo {
  SmallVector<int64_t> sizes;
  Value memref;
  SmallVector<int64_t> shape;

  TileInfo() {}
  TileInfo(SmallVector<int64_t> tileSizes, Value memref)
      : sizes{tileSizes.begin(), tileSizes.end()}, memref{memref}, shape{} {
    auto memShape = memref.getType().cast<MemRefType>().getShape();
    shape = {memShape.begin(), memShape.end()};
  }

  SmallVector<int64_t> getTileDims() const {
    SmallVector<int64_t> tileDims(sizes.size());
    for (unsigned i = 0; i < tileDims.size(); ++i)
      tileDims[i] = (int64_t)ceil((double)shape[i] / sizes[i]);
    return tileDims;
  }
};
} // namespace

/// Tile the input MemRefType statically.
/// If passed tileOnly = true, we won't include the partition dims.
static MemRefType getTiledMemRefType(const TileInfo &tileInfo,
                                     bool tileOnly = false,
                                     bool flatten = false) {
  MemRefType src = tileInfo.memref.getType().cast<MemRefType>();
  assert(src.getLayout().isIdentity() &&
         "We don't support memref with non-identity affine map.");

  SmallVector<int64_t> dstShape;
  auto srcShape = src.getShape();
  if (!tileOnly) {
    if (!flatten)
      for (unsigned i = 0; i < srcShape.size(); ++i)
        dstShape.push_back(
            (int64_t)ceil((double)srcShape[i] / tileInfo.sizes[i]));
    else {
      int64_t dim = 1;
      for (unsigned i = 0; i < srcShape.size(); ++i)
        dim *= (int64_t)ceil((double)srcShape[i] / tileInfo.sizes[i]);
      dstShape.push_back(dim);
    }
  }

  for (unsigned i = 0; i < srcShape.size(); ++i)
    dstShape.push_back(tileInfo.sizes[i]);

  return MemRefType::get(dstShape, src.getElementType());
}

/// Check if each result is an dim/symbol expression. Make sure they are all
/// distinct.
static bool isIdentityAffineMap(const AffineMap &affMap) {
  if (affMap.getNumDims() + affMap.getNumSymbols() != affMap.getNumResults())
    return false;
  ArrayRef<AffineExpr> results = affMap.getResults();

  LLVM_DEBUG({
    dbgs() << "Checking if ";
    affMap.dump();
    dbgs() << " is identity map or not\n";
  });

  SmallSetVector<unsigned, 4> dimVis, symbolVis;
  for (unsigned i = 0; i < affMap.getNumResults(); ++i) {
    if (auto expr = results[i].dyn_cast<AffineDimExpr>()) {
      if (dimVis.count(i))
        return false;
      dimVis.insert(expr.getPosition());
    } else if (auto expr = results[i].dyn_cast<AffineSymbolExpr>()) {
      if (symbolVis.count(i))
        return false;
      symbolVis.insert(expr.getPosition());
    } else {
      // No other kind of expressions.
      return false;
    }
  }
  return true;
}

static MapVector<Value, TileInfo> getTilingInfo(ArrayRef<Value> memrefs,
                                                ModuleOp m) {
  MapVector<Value, TileInfo> tiling;
  // See if they have simple access patterns that can be directly extracted.
  for (Value memref : memrefs) {
    LLVM_DEBUG({
      dbgs() << "Trying to tile: ";
      memref.dump();
    });
    // Check if all the users of memref are phism.pe callers.
    if (any_of(memref.getUsers(), [&](Operation *op) {
          return !isa<CallOp>(op) || !op->hasAttr("phism.pe");
        })) {
      LLVM_DEBUG({
        memref.dump();
        dbgs() << " has been skipped since it has non PE caller users.\n";
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
      dbgs() << "Found the following accesses:\n";
      for (MemRefAccess &access : accesses)
        access.opInst->dump();
      dbgs() << "---------------------------\n";
    });

    // Check if all accesses are idenity maps.
    if (any_of(accesses, [&](MemRefAccess &access) {
          AffineValueMap vmap;
          access.getAccessMap(&vmap);
          return !isIdentityAffineMap(vmap.getAffineMap());
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
    // Map from access index -> domain
    MapVector<unsigned, AffineMap> lbMaps, ubMaps;

    auto isSame = [&](const MapVector<unsigned, AffineMap> &ma,
                      const MapVector<unsigned, AffineMap> &mb) -> bool {
      for (const auto &it : ma) {
        auto it2 = mb.find(it.first);
        if (it2 == mb.end())
          return false;
        if (it2->second != it.second)
          return false;
      }
      return true;
    };

    for (MemRefAccess &ma : accesses) {
      AffineValueMap avm;
      ma.getAccessMap(&avm);

      MapVector<unsigned, AffineForOp> forOps;

      // Based on the presumption that every result should be an unique operand.
      bool hasInvalidAddr = false;
      for (unsigned i = 0; i < avm.getNumResults(); ++i) {
        // i is the index to a dim in the address.
        AffineExpr result = avm.getResult(i);
        unsigned index = avm.getNumResults();
        if (auto expr = result.dyn_cast<AffineDimExpr>())
          index = expr.getPosition();
        else if (auto expr = result.dyn_cast<AffineSymbolExpr>())
          index = expr.getPosition() + avm.getNumDims();
        assert(index != avm.getNumResults() &&
               "The presumption on identity mapping is not satisfied.");

        Value operand = avm.getOperand(index);
        if (isForInductionVar(operand))
          forOps[i] = getForInductionVarOwner(operand);
        else {
          LLVM_DEBUG(dbgs()
                     << "Address index #" << i << " has been skipped.\n");
          hasInvalidAddr = true;
          break;
        }
      }

      if (hasInvalidAddr) {
        isIdentical = false;
        break;
      }

      MapVector<unsigned, AffineMap> tmpLbMaps, tmpUbMaps;
      for (auto p : forOps) {
        AffineForOp forOp;
        unsigned index;
        std::tie(index, forOp) = p;

        // Filter out the result that are constants. We don't care about them.
        // ()[s0] -> (70, s0 * 32 + 32) will be ()[s0] -> (s0 * 32 + 32)
        AffineMap lbMap = filterExtraConstantResults(forOp.getLowerBoundMap());
        AffineMap ubMap = filterExtraConstantResults(forOp.getUpperBoundMap());

        if (lbMap.isSingleConstant() && ubMap.isSingleConstant()) {
          llvm::errs() << "There appears a pair of constant loop bounds. We "
                          "cannot deal with this yet.\n";
          isIdentical = false;
          break;
        }

        tmpLbMaps[index] = lbMap;
        tmpUbMaps[index] = ubMap;
      }

      if (!isIdentical)
        break;

      // Simply ignore those with constant lower upper bounds.
      // They won't cause much trouble (heuristically) if we don't partition
      // for them.
      if (any_of(
              tmpLbMaps,
              [&](const auto &it) { return it.second.isSingleConstant(); }) ||
          any_of(tmpUbMaps, [&](const auto &it) {
            return it.second.isSingleConstant();
          })) {
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
        isIdentical &= isSame(tmpLbMaps, lbMaps) && isSame(tmpUbMaps, ubMaps);
        if (!isIdentical) {
          LLVM_DEBUG(dbgs() << "Found not identical loop bound maps.\n");
          break;
        }
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
    if (any_of(lbMaps,
               [&](const auto &it) { return it.second.getNumResults() > 1; }) ||
        any_of(ubMaps,
               [&](const auto &it) { return it.second.getNumResults() > 1; })) {
      LLVM_DEBUG(llvm::errs() << "There are loop bounds have more than one "
                                 "non-constant expressions.\n";);
      continue;
    }
    // Check whether the loop bounds satisfy the tiling constraints.
    if (any_of(lbMaps,
               [&](const auto &it) { return !isTiledLoopBound(it.second); }) ||
        any_of(ubMaps,
               [&](const auto &it) { return !isTiledLoopBound(it.second); })) {
      LLVM_DEBUG(llvm::errs() << "Loop bounds are not tiled expressions.\n";);
      continue;
    }

    // Finally resolve the tile size.
    MapVector<unsigned, int64_t> tileSizeMap;
    for (const auto &it : lbMaps) {
      int64_t tileSize = getTileSize(it.second);
      if (tileSize != getTileSize(ubMaps[it.first]))
        continue;
      tileSizeMap[it.first] = tileSize;
    }

    // For those cannot be tiled, set the tile size equal to the dim size.
    MemRefType ty = memref.getType().cast<MemRefType>();
    for (unsigned i = 0; i < ty.getRank(); ++i) {
      if (lbMaps.count(i))
        continue;
      tileSizeMap[i] = ty.getShape()[i];
    }

    // Abandon further processing if the tile size cannot match memref's type.
    if ((int64_t)tileSizeMap.size() != ty.getRank()) {
      LLVM_DEBUG(
          dbgs() << "Tile sizes are not equal to the rank of the memref.\n");
      continue;
    }

    SmallVector<int64_t> tileSizes(ty.getRank());
    for (unsigned i = 0; i < ty.getRank(); ++i)
      tileSizes[i] = tileSizeMap[i];

    // The resolved memref tiling.
    LLVM_DEBUG({
      dbgs() << "Memref ";
      memref.dump();
      dbgs() << " has been tiled into: ";
      interleaveComma(tileSizes, dbgs());
      dbgs() << "\n\n";
    });
    tiling[memref] = TileInfo{tileSizes, memref};
  }

  return tiling;
}

static FuncOp createTiledCallee(Value memref, mlir::FuncOp callee,
                                mlir::CallOp caller, const TileInfo &tileInfo,
                                bool flatten, unsigned stage, OpBuilder &b) {
  // Get the type of a MemRef tile.
  MemRefType newMemRefType =
      getTiledMemRefType(tileInfo, /*tileOnly=*/true, /*flatten=*/flatten);

  LLVM_DEBUG({
    dbgs() << " * New memref type: ";
    newMemRefType.dump();
    dbgs() << "\n";
  });

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
  assert(memId < caller.getNumOperands());

  // New callee function type.
  FunctionType newFuncType =
      b.getFunctionType(newArgTypes, callee->getResultTypes());
  b.setInsertionPointAfter(callee);
  FuncOp newCallee = b.create<FuncOp>(
      callee.getLoc(),
      std::string(callee.getName()) + "_" + std::to_string(stage), newFuncType);

  newCallee->setAttr("phism.pe", b.getUnitAttr());

  Block *entry = newCallee.addEntryBlock();
  b.setInsertionPointToEnd(entry);
  b.create<mlir::ReturnOp>(callee.getLoc());
  LLVM_DEBUG({
    dbgs() << " * New callee created (body empty):\n";
    newCallee.dump();
  });

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

  LLVM_DEBUG(dbgs() << "------> Rewriting the body of the new callee.\n");
  // Rewrite the loop iterators for memory accesses.
  // For now I think the new iterator should be %i mod tile_size.
  // So we would simply create the corresponding new iterators, and use
  // them to replace the old ones applied to the tiled memref.
  newCallee.walk([&](Operation *op) {
    if (!isa<AffineLoadOp, AffineStoreOp>(op))
      return;

    LLVM_DEBUG({
      dbgs() << "---> Working on: ";
      op->dump();
    });
    // This affine.load/store op should have accessed the target memref.
    if (find(op->getOperands(), newCallee.getArgument(memId)) ==
        op->operand_end())
      return;

    b.setInsertionPoint(op);

    MemRefAccess access(op);
    AffineValueMap vmap;
    access.getAccessMap(&vmap);

    for (unsigned dim = 0; dim < vmap.getNumResults(); ++dim) {
      AffineExpr result = vmap.getResult(dim);

      unsigned index = vmap.getNumOperands();
      if (auto expr = result.dyn_cast<AffineDimExpr>())
        index = expr.getPosition();
      else if (auto expr = result.dyn_cast<AffineSymbolExpr>())
        index = expr.getPosition() + vmap.getNumDims();
      assert(index != vmap.getNumOperands());

      Value operand = vmap.getOperand(index);
      if (!isForInductionVar(operand))
        continue;

      // The affine map that does the modulo operation.
      AffineExpr modExpr =
          b.getAffineDimExpr(0) % b.getAffineConstantExpr(tileInfo.sizes[dim]);
      AffineMap affMap = AffineMap::get(1, 0, modExpr);
      AffineApplyOp newInd =
          b.create<AffineApplyOp>(op->getLoc(), affMap, operand);

      // Note that this is the operand of the op, not the vmap. So we should use
      // the shifted dim, not index.
      unsigned pos = dim + (isa<AffineLoadOp>(op) ? 1 : 2);
      op->setOperand(pos, newInd);
    }
  });

  return newCallee;
}

static bool isAffineLoadOrStoreOnTargetMemRef(Operation *op, Value memref) {
  if (!isa<mlir::AffineLoadOp, AffineStoreOp>(op))
    return false;
  if (mlir::AffineLoadOp loadOp = dyn_cast<mlir::AffineLoadOp>(op))
    if (loadOp.getMemRef() != memref)
      return false;
  if (mlir::AffineStoreOp storeOp = dyn_cast<mlir::AffineStoreOp>(op))
    if (storeOp.getMemRef() != memref)
      return false;
  return true;
}

static SmallVector<Value> resolveTileIndices(Operation *op, Value memref) {
  if (!isAffineLoadOrStoreOnTargetMemRef(op, memref)) {
    LLVM_DEBUG(
        dbgs()
        << "-> Skipped operation of type " << op->getName()
        << " since it is not an affine.load/store on the target memref.\n");
    return {};
  }

  LLVM_DEBUG({
    dbgs() << " * Found affine.load/store operation: ";
    op->dump();
  });

  SmallVector<Value> indices;
  // The first operand for affine.store would be the value to be stored.
  unsigned addrStartIdx = isa<mlir::AffineLoadOp>(op) ? 1 : 2;

  for (unsigned i = addrStartIdx; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);

    // The index for a tiled memref will be from an affine.apply op.
    mlir::AffineApplyOp applyOp = operand.getDefiningOp<mlir::AffineApplyOp>();
    if (!applyOp) {
      LLVM_DEBUG({
        dbgs() << " * The " << i
               << "-th operand is not from an affine.apply op: ";
        operand.dump();
        dbgs() << "\n";
      });

      if (operand.isa<BlockArgument>() &&
          isa<FuncOp>(operand.getParentBlock()->getParentOp())) {
        LLVM_DEBUG(dbgs() << "It is a function argument. Kept as an index.\n");
        indices.push_back(operand);
      }

      continue;
    }
    LLVM_DEBUG({
      dbgs() << " * Got affine.apply op for index " << i << ": ";
      operand.dump();
    });

    // Constraints on the apply operation.
    assert(applyOp.getNumOperands() == 1);
    assert(applyOp.getAffineMap().getNumResults() == 1);
    assert(applyOp.getAffineMap().getResult(0).getKind() ==
           AffineExprKind::Mod);

    Value indvar = applyOp.getOperand(0);
    // At least one bound should have a single operand (for the loop
    // indvar).
    mlir::AffineForOp forOp = getForInductionVarOwner(indvar);
    LLVM_DEBUG(dbgs() << "   * Lower bound: " << forOp.getLowerBoundMap()
                      << "\n");
    LLVM_DEBUG(dbgs() << "   * Upper bound: " << forOp.getUpperBoundMap()
                      << "\n");

    if (!(forOp.getLowerBoundOperands().size() == 1 ||
          forOp.getUpperBoundOperands().size() == 1))
      continue;

    Value source = forOp.getUpperBoundOperands().size() == 1
                       ? forOp.getUpperBoundOperands()[0]
                       : forOp.getLowerBoundOperands()[0];
    assert(forOp.getLowerBoundOperands().size() < 1 ||
           source == forOp.getLowerBoundOperands()[0]);

    LLVM_DEBUG(dbgs() << "   * Top-level index: " << source << '\n');
    indices.push_back(source);
  }

  return indices;
}

static Value createSubViewOfTiledMemRefWithFullRank(
    ArrayRef<Value> indices, Value tiledMemRef, unsigned rank,
    const BlockAndValueMapping &vmap, OpBuilder &b) {
  assert(indices.size() == rank / 2 &&
         "The size of the tile indices should be the same as rank / 2.");
  SmallVector<OpFoldResult> offsets, sizes, strides;

  auto memRefType = tiledMemRef.getType().cast<MemRefType>();

  // Figure out the offsets.
  // Need to know which tile loop has used by the accessed memref.
  for (unsigned i = 0; i < rank / 2; ++i) {
    Value index = vmap.lookup(indices[i]);
    if (index.isa<BlockArgument>() &&
        isa<FuncOp>(index.getParentBlock()->getParentOp()))
      offsets.push_back(b.getIndexAttr(0));
    else
      offsets.push_back(index);
  }
  for (unsigned i = 0; i < rank / 2; ++i)
    offsets.push_back(b.getIndexAttr(0));

  // Figure out sizes.
  for (unsigned i = 0; i < rank / 2; ++i)
    sizes.push_back(b.getIndexAttr(1));
  for (unsigned i = 0; i < rank / 2; ++i)
    sizes.push_back(b.getIndexAttr(memRefType.getShape()[i + rank / 2]));

  // Figure out strides.
  for (unsigned i = 0; i < rank; ++i)
    strides.push_back(b.getIndexAttr(1));

  // Get the resulting type for the subview. Otherwise, it won't
  // match.
  MemRefType newTiledMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          rank / 2, memRefType.cast<MemRefType>(), offsets, sizes, strides)
          .cast<MemRefType>();

  // The final subview operaion.
  memref::SubViewOp subView =
      b.create<memref::SubViewOp>(tiledMemRef.getLoc(), newTiledMemRefType,
                                  tiledMemRef, offsets, sizes, strides);

  // Strip the affine map
  MemRefType castMemRefType = MemRefType::get(newTiledMemRefType.getShape(),
                                              memRefType.getElementType());
  memref::CastOp cast =
      b.create<memref::CastOp>(tiledMemRef.getLoc(), subView, castMemRefType);

  return cast;
}

static Value createSubViewOfTiledMemRefWithFlattenedRank(
    ArrayRef<Value> indices, Value tiledMemRef, unsigned rank,
    const BlockAndValueMapping &vmap, const TileInfo &tileInfo, OpBuilder &b) {
  assert(indices.size() == rank - 1 &&
         "The size of the tile indices should be the same as rank - 1.");
  SmallVector<OpFoldResult> offsets, sizes, strides;

  auto memRefType = tiledMemRef.getType().cast<MemRefType>();

  // Create the flattened tile dims.
  auto tileDims = tileInfo.getTileDims();

  SmallVector<int64_t> parDims(tileDims.size());
  parDims[tileDims.size() - 1] = 1;
  for (unsigned i = 1; i < tileDims.size(); ++i)
    parDims[tileDims.size() - i - 1] =
        parDims[tileDims.size() - i] * tileDims[tileDims.size() - i];

  AffineExpr indexExpr = b.getAffineConstantExpr(0);
  for (unsigned i = 0; i < parDims.size(); ++i) {
    Value index = vmap.lookup(indices[i]);

    // The dim will be ignored if it is a function arg.
    if (index.isa<BlockArgument>() &&
        isa<FuncOp>(index.getParentBlock()->getParentOp()))
      continue;

    indexExpr =
        indexExpr + b.getAffineConstantExpr(parDims[i]) * b.getAffineDimExpr(i);
  }
  SmallVector<Value> mappedIndices;
  for (Value idx : indices)
    mappedIndices.push_back(vmap.lookup(idx));
  Value index = b.create<AffineApplyOp>(
      tiledMemRef.getLoc(), AffineMap::get(parDims.size(), 0, indexExpr),
      mappedIndices);

  // Figure out the offsets.
  // Need to know which tile loop has used by the accessed memref.
  offsets.push_back(index);
  for (unsigned i = 0; i < rank - 1; ++i)
    offsets.push_back(b.getIndexAttr(0));

  // Figure out sizes.
  sizes.push_back(b.getIndexAttr(1));
  for (unsigned i = 0; i < rank - 1; ++i)
    sizes.push_back(b.getIndexAttr(memRefType.getShape()[i + 1]));

  // Figure out strides.
  for (unsigned i = 0; i < rank; ++i)
    strides.push_back(b.getIndexAttr(1));

  // Get the resulting type for the subview. Otherwise, it won't
  // match.
  MemRefType newTiledMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          rank - 1, memRefType.cast<MemRefType>(), offsets, sizes, strides)
          .cast<MemRefType>();

  // The final subview operaion.
  memref::SubViewOp subView =
      b.create<memref::SubViewOp>(tiledMemRef.getLoc(), newTiledMemRefType,
                                  tiledMemRef, offsets, sizes, strides);

  // Strip the affine map
  MemRefType castMemRefType = MemRefType::get(newTiledMemRefType.getShape(),
                                              memRefType.getElementType());
  memref::CastOp cast =
      b.create<memref::CastOp>(tiledMemRef.getLoc(), subView, castMemRefType);

  return cast;
}

static Value buildSubViewForTiledMemRef(FuncOp tiledCallee, mlir::CallOp caller,
                                        Value tiledMemRef, unsigned argIdx,
                                        bool flatten, const TileInfo &tileInfo,
                                        OpBuilder &b) {
  LLVM_DEBUG({
    dbgs() << "\n---> Building memref.subview of ";
    tiledMemRef.dump();
  });

  // Get the static rank.
  MemRefType tiledMemRefType = tiledMemRef.getType().cast<MemRefType>();
  unsigned rank = tiledMemRefType.getRank();

  LLVM_DEBUG(dbgs() << "-> Iterate every op in the tiled callee to find the "
                       "correct subview indices for memref type "
                    << tiledMemRefType << ".\n");
  // Look into the callee to find which tile loop has been used to
  // access the corresponding tiled memory dimensions.
  SmallVector<Value> indices;
  tiledCallee.walk([&](Operation *op) {
    auto indices_ = resolveTileIndices(op, tiledCallee.getArgument(argIdx));
    if (indices_.empty())
      return;

    LLVM_DEBUG({
      dbgs() << " * Resolved indices:\n";
      for (auto ind : enumerate(indices_))
        dbgs() << '\t' << ind.index() << ": " << ind.value() << '\n';
      dbgs() << '\n';
    });

    if (indices.empty()) {
      LLVM_DEBUG(
          dbgs() << "-> Initialise the final indices by the current ones.\n");
      std::swap(indices_, indices);
    } else {
      LLVM_DEBUG({
        if (indices_ != indices) {
          dbgs() << "-> Currently resolved indices don't match the aggregated "
                    "ones.\n";
          dbgs() << "Currently resolved:\n";
          for (auto ind : enumerate(indices_))
            dbgs() << '\t' << ind.index() << ": " << ind.value();
          dbgs() << '\n';
          dbgs() << "Aggregated:\n";
          for (auto ind : enumerate(indices))
            dbgs() << '\t' << ind.index() << ": " << ind.value();
          dbgs() << '\n';
        } else {
          dbgs() << " * Matches previously resolved indices.\n";
        }
      });
      assert(indices_ == indices &&
             "Currently resolved indices should match the aggregated ones.");

      std::swap(indices_, indices);
    }
  });

  BlockAndValueMapping vmap;
  vmap.map(tiledCallee.getArguments(), caller.getArgOperands());

  b.setInsertionPoint(caller);
  if (flatten)
    return createSubViewOfTiledMemRefWithFlattenedRank(indices, tiledMemRef,
                                                       rank, vmap, tileInfo, b);
  return createSubViewOfTiledMemRefWithFullRank(indices, tiledMemRef, rank,
                                                vmap, b);
}

static FuncOp tileTopFunction(FuncOp top, ArrayRef<Value> memrefs,
                              MapVector<Value, TileInfo> &tiling, bool flatten,
                              ModuleOp m, OpBuilder &b) {
  LLVM_DEBUG(
      dbgs() << "\n===================================================\n\n"
             << " * Tiling the top function: " << top.getName()
             << " with flatten set to " << flatten << ".\n\n");

  // Things to process.
  SmallVector<Value> worklist;
  for (auto &it : tiling)
    worklist.push_back(it.first);

  // Next, we will resolve the memory tiling one by one -
  FuncOp prevFunc = top;

  for (unsigned stage = 0; stage < worklist.size(); ++stage) {
    LLVM_DEBUG(dbgs() << "\n------> Processing stage: " << stage << "\n\n");

    // Rebuild the caller list.
    SmallVector<mlir::CallOp> callers;
    prevFunc.walk([&](mlir::CallOp caller) {
      if (caller->hasAttr("phism.pe"))
        callers.push_back(caller);
    });

    Value memref = worklist[stage];
    TileInfo tileInfo = tiling[memref];

    LLVM_DEBUG({
      dbgs() << " * Tiling memref:\n";
      memref.dump();
      dbgs() << " * Resulting tile sizes: [";
      interleaveComma(tileInfo.sizes, dbgs());
      dbgs() << "]\n";
    });

    // -------------------------------------------------------------------
    // Step 1: create a function of with an interface of the tiled input.
    MemRefType newMemRefType =
        getTiledMemRefType(tileInfo, /*tileOnly=*/false, /*flatten=*/flatten);
    LLVM_DEBUG({
      llvm::errs() << " * New MemRef type: ";
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
      llvm::errs() << " * New function type: ";
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

    LLVM_DEBUG({
      dbgs() << " * New function created (body empty):\n";
      newFunc.dump();
    });

    // Map from the old callee to the new one.
    SmallDenseMap<FuncOp, FuncOp> calleeMap;

    LLVM_DEBUG(dbgs() << "------> Creating new caller/callees.\n");

    // Create the __tiled version for each PE that has been affected by the
    // tiling, i.e., uses the memref.
    for (mlir::CallOp caller : callers) {
      // This caller should use the target memref.
      if (find(caller.getArgOperands(), memref) == caller.arg_operand_end()) {
        LLVM_DEBUG({
          dbgs() << "------> Skipped caller: ";
          caller.dump();
          dbgs() << " since it doesn't use the target memref.\n";
        });
        continue;
      }

      LLVM_DEBUG({
        dbgs() << "------> Working on caller: ";
        caller.dump();
      });

      FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
      LLVM_DEBUG({
        dbgs() << " * Found old callee: \n";
        callee.dump();
      });

      // Finalise the result to the map.
      FuncOp newCallee = createTiledCallee(memref, callee, caller, tileInfo,
                                           flatten, stage, b);

      LLVM_DEBUG({
        dbgs() << " * Created new callee: \n";
        newCallee.dump();
      });

      calleeMap.insert({callee, newCallee});
    }

    LLVM_DEBUG({
      dbgs() << "\n\n";
      dbgs() << " * Callee mapping results:\n";
      for (auto &it : calleeMap)
        dbgs() << "\t" << it.first.getName() << "\t==> " << it.second.getName()
               << "\n";
    });

    LLVM_DEBUG({
      dbgs() << "\n\n";
      dbgs() << "------> Filling function body ...\n\n";
    });

    // Argument map.
    BlockAndValueMapping vmap;
    vmap.map(prevFunc.getArguments(), newFunc.getArguments());

    LLVM_DEBUG({
      dbgs() << "* Argument mappings:\n";
      for (Value &arg : prevFunc.getArguments())
        dbgs() << "\t" << arg.getType() << "\t==> "
               << vmap.lookup(arg).getType() << "\n";
      dbgs() << "\n";
    });

    // Iterate every operation in the original callee and clone it to the
    // new one.
    b.setInsertionPointToStart(entry);
    for (Operation &op : prevFunc.getBlocks().begin()->getOperations()) {
      if (isa<mlir::ReturnOp>(op))
        continue;
      b.clone(op, vmap);
    }

    SmallVector<mlir::CallOp> toRemove;

    LLVM_DEBUG(dbgs() << "------> Replacing memref ARGS in cloned callers.\n");
    // Replace the callers in the cloned function.
    newFunc.walk([&](mlir::CallOp caller) {
      // Try to see if the caller reaches to the tiled version.
      FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
      if (!calleeMap.count(callee)) // not affected by tiling
        return;

      LLVM_DEBUG({
        dbgs() << "---> Replacing caller: ";
        caller.dump();
      });

      // Now we replace the original caller to call the new callee, with the
      // memref argument replaced.
      FuncOp newCallee = calleeMap[callee];

      // The new memref should be found.
      Value newMemRef = vmap.lookup(memref);

      SmallVector<Value> args;
      for (auto arg : enumerate(caller.getArgOperands())) {
        if (arg.value() != newMemRef)
          args.push_back(arg.value());
        else {
          b.setInsertionPointAfter(caller);
          Value subView = buildSubViewForTiledMemRef(
              newCallee, caller, newMemRef, arg.index(), flatten, tileInfo, b);
          LLVM_DEBUG(dbgs() << " * Created subview: "
                            << subView.getDefiningOp()->getOperand(0) << '\n');
          LLVM_DEBUG(dbgs() << " * Created cast: " << subView << '\n');
          args.push_back(subView);
        }
      }

      // Note that we still use the original callee symbol here.
      mlir::CallOp newCaller =
          b.create<mlir::CallOp>(caller.getLoc(), newCallee, args);
      newCaller->setAttr("phism.pe", b.getUnitAttr());

      LLVM_DEBUG(dbgs() << " * New caller: " << newCaller << '\n');

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

    prevFunc = newFunc;

    LLVM_DEBUG(dbgs() << "------> Finished stage: " << stage
                      << ", created: \n\n"
                      << newFunc << "\n\n");
  }

  return prevFunc;
}

/// Put all the visited functions into a set.
static void markCalledFunctions(FuncOp top, ModuleOp m,
                                SmallPtrSet<FuncOp, 4> &visited) {
  visited.insert(top);

  top.walk([&](CallOp caller) {
    visited.insert(cast<FuncOp>(m.lookupSymbol(caller.getCallee())));
  });
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
                      SymbolRefAttr::get(caller.getContext(),
                                         newNames[caller.getCallee()]));
  });
}

namespace {

struct SimpleArrayPartitionPass
    : public phism::SimpleArrayPartitionBase<SimpleArrayPartitionPass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // Get the top function.
    FuncOp top = findPhismTop(m);
    if (!top) {
      m.emitRemark() << "No top function found for array partition. Have you "
                        "forgot to annotate {phism.pe} to callers?\n";
      return;
    }

    // Before transformation, keep all the existing functions into a set so that
    // they won't be recycled later.
    SmallPtrSet<FuncOp, 4> keep;
    getFunctionsToKeep(m, top, keep);

    SmallVector<CallOp> callers;
    top.walk([&](CallOp caller) {
      if (caller->hasAttr("phism.pe"))
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
    for (Value memref : memrefs)
      if (!tiling.count(memref)) {
        LLVM_DEBUG({
          dbgs() << "There is at least one memref: ";
          memref.dump();
          dbgs() << " has not partitioned. We discard the whole case since the "
                    "performance gain would be minor.\n";
        });
        return;
      }

    auto tilingCopy = tiling;

    // Tile the top function.
    FuncOp newTop = tileTopFunction(top, memrefs, tiling, flatten, m, b);
    // newTop->setAttrs(top->getAttrs());

    LLVM_DEBUG(dbgs() << "------> Clean up created auxilliary functions.\n");

    // Clean up.
    markCalledFunctions(newTop, m, keep);
    sweepUncalledFunctions(m, keep);

    // Reset names.
    renameTiledFunctions(m, b);

    // If array partition has been succesful, dump a file that stores the
    // corresponding information.
    if (dumpFile) {
      LLVM_DEBUG(dbgs() << "------> Dump file to array_partition.txt.\n");
      std::ofstream infoFile;
      infoFile.open("array_partition.txt", std::ios::out);
      if (infoFile.is_open()) {
        for (auto &it : tilingCopy) {
          interleave(
              it.second.sizes,
              [&](const int64_t &size) { infoFile << std::to_string(size); },
              [&]() { infoFile << ", "; });
          infoFile << '\n';
        }
      }
    }

    if (genMain) {
      FuncOp orig = dyn_cast<FuncOp>(m.lookupSymbol("main"));
      if (orig)
        orig.erase();

      Location loc = newTop.getLoc();
      b.setInsertionPointAfter(newTop);

      StringRef name = "main";
      FunctionType ty = b.getFunctionType({}, {b.getIntegerType(32)});
      FuncOp main = b.create<FuncOp>(loc, name, ty);

      Block *entry = main.addEntryBlock();
      b.setInsertionPointToStart(entry);

      b.create<mlir::ReturnOp>(
          loc, b.create<arith::ConstantIntOp>(loc, 0, 32).getResult());

      // Iterate every top argument.
      b.setInsertionPointToStart(entry);

      SmallVector<Value> operands;
      for (auto it : enumerate(newTop.getArguments())) {
        Value arg = it.value();
        unsigned index = it.index();
        Type ty = arg.getType();
        if (auto intTy = ty.dyn_cast<IntegerType>()) {
          IntegerAttr attr = newTop.getArgAttr(index, "scop.constant_value")
                                 .dyn_cast<IntegerAttr>();
          Value operand = b.create<arith::ConstantIntOp>(loc, attr.getInt(), 32)
                              .getResult();
          operands.push_back(operand);
        } else if (auto floatTy = ty.dyn_cast<FloatType>()) {
          Value operand = b.create<arith::ConstantFloatOp>(
              loc, llvm::APFloat(1.0), floatTy);
          operands.push_back(operand);
        } else if (auto memTy = ty.dyn_cast<MemRefType>()) {
          Value operand = b.create<memref::AllocaOp>(loc, memTy);
          operands.push_back(operand);
        }
      }

      LLVM_DEBUG({
        dbgs() << "List of operands:\n";
        for (Value operand : operands)
          dbgs() << " * " << operand << "\n";
      });

      b.create<CallOp>(loc, newTop, operands);
    }
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createSimpleArrayPartitionPass() {
  return std::make_unique<SimpleArrayPartitionPass>();
}
