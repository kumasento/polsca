//===- LiftMemRefSubview.cc - simplify access ------------------ C++-===//

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

#define DEBUG_TYPE "lift-subview"

using namespace mlir;
using namespace llvm;
using namespace phism;

namespace {
struct MemRefLiftInfo {
  Value memref;
  unsigned index; // the function argument index.
  MemRefType type;
  SmallVector<Value> indices;
  MLIRContext *ctx;

  MemRefLiftInfo(Value memref, unsigned index)
      : memref{memref}, index{index},
        type{memref.getType().dyn_cast<MemRefType>()},
        ctx{memref.getContext()} {}

  MemRefType getLiftedType() const;
};

/// This type has no layout information. It could be RISKY.
MemRefType MemRefLiftInfo::getLiftedType() const {
  return MemRefType::Builder(type.getShape().drop_front(indices.size()),
                             type.getElementType());
}

} // namespace

static bool getIndices(Operation *op, SmallVectorImpl<Value> &indices) {
  indices.clear();
  if (isa<mlir::AffineLoadOp, mlir::AffineStoreOp>(op)) {
    MemRefAccess access(op);

    AffineValueMap avm;
    access.getAccessMap(&avm);

    for (unsigned i = 0; i < avm.getNumResults(); ++i) {
      AffineExpr expr = avm.getResult(i);
      if (auto symbol = expr.dyn_cast<AffineSymbolExpr>())
        indices.push_back(
            avm.getOperand(symbol.getPosition() + avm.getNumDims()));
    }

    return true;
  } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
    indices.append(op->operand_begin() + (isa<memref::LoadOp>(op) ? 1 : 2),
                   op->operand_end());
    return true;
  }
  return false;
}

/// Make sure each memref has the same access pattern.
static void resolveOffset(llvm::MapVector<Value, MemRefLiftInfo> &memrefs) {
  for (auto it = memrefs.begin(); it != memrefs.end(); ++it) {
    Value memref = it->first;
    MemRefLiftInfo &info = it->second;

    if (memref.use_empty()) // There is no use, offset is by default 0.
      continue;

    bool started = false;

    // Iterate every access.
    for (Operation *op : memref.getUsers()) {
      SmallVector<Value> indices;
      if (!getIndices(op, indices)) // not an access.
        continue;

      LLVM_DEBUG({
        dbgs() << " * Access: " << (*op) << "\n";
        dbgs() << " * Found indices: \n";
        for (auto p : enumerate(indices))
          dbgs() << "    #" << p.index() << ": " << p.value() << '\n';
      });

      // not initialised, just use this access's indices.
      if (!started) {
        std::swap(info.indices, indices);
        started = true;
        continue;
      }

      if (info.indices.empty())
        continue;

      // figure out the new offset and pop the back.
      if (indices.empty())
        info.indices = {};
      else {
        for (unsigned i = 0, e = std::min(info.indices.size(), indices.size());
             i < e; ++i) {
          if (info.indices[i] != indices[i]) {
            // New offset determined.
            info.indices.pop_back_n(info.indices.size() - i);
            break;
          }
        }
      }
    }

    // Finally, remove all the indices that are not from the block arguments of
    // the parent function.
    unsigned keep = 0;
    for (auto index : info.indices) {
      if (!index.isa<BlockArgument>())
        break;
      auto arg = index.dyn_cast<BlockArgument>();
      if (!isa<FuncOp>(arg.getOwner()->getParentOp()))
        break;

      ++keep;
    }
    info.indices.pop_back_n(info.indices.size() - keep);
  }

  LLVM_DEBUG({
    dbgs() << "Resolved offset:\n";
    for (auto &it : memrefs) {
      dbgs() << " * Memref:  " << it.first << "\n";
      dbgs() << " * Indices: \n";
      for (auto p : enumerate(it.second.indices)) {
        dbgs() << "    #" << p.index() << ": " << p.value() << '\n';
      }
      dbgs() << "\n";
    }
  });
}

static FunctionType replaceArgumentType(FuncOp f, unsigned index,
                                        Type newType) {
  FunctionType fty = f.getType();

  SmallVector<Type> argTypes;
  for (Type ty : fty.getInputs())
    argTypes.push_back(ty);

  argTypes[index] = newType;

  return FunctionType::get(f.getContext(), argTypes, fty.getResults());
}

static LogicalResult liftCallee(FuncOp callee, MemRefLiftInfo &info) {
  Value memref = info.memref;
  LLVM_DEBUG(dbgs() << "\n ---> Lifting memref: " << memref << "\n");

  // Get the new memref type.
  MemRefType liftedType = info.getLiftedType();
  LLVM_DEBUG(dbgs() << " * Lifted type: " << liftedType << '\n');

  // Replace the function argument type.
  FunctionType fty = replaceArgumentType(callee, info.index, liftedType);
  LLVM_DEBUG(dbgs() << " * New callee type: " << fty << '\n');
  callee.setType(fty);

  // Update entry block argument type.
  Block *entry = &callee.getBlocks().front();
  entry->getArgument(info.index).setType(liftedType);

  // Update every accesses
  for (Operation *user : memref.getUsers()) {
    if (isa<memref::LoadOp, memref::StoreOp>(user)) {
      // Just pop out the indices
      unsigned start = isa<memref::LoadOp>(user) ? 1 : 2;
      user->eraseOperands(start, info.indices.size());
    } else if (isa<mlir::AffineLoadOp, mlir::AffineStoreOp>(user)) {
      // Need to update the affine map.
      if (auto loadOp = dyn_cast<mlir::AffineLoadOp>(user)) {
        AffineMap am = loadOp.getAffineMap();
        auto results = am.getResults().drop_front(info.indices.size());
        loadOp->setAttr(loadOp.getMapAttrName(),
                        AffineMapAttr::get(
                            AffineMap::get(am.getNumDims(), am.getNumSymbols(),
                                           results, loadOp.getContext())));
      } else if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(user)) {
        AffineMap am = storeOp.getAffineMap();
        auto results = am.getResults().drop_front(info.indices.size());
        storeOp->setAttr(storeOp.getMapAttrName(),
                         AffineMapAttr::get(
                             AffineMap::get(am.getNumDims(), am.getNumSymbols(),
                                            results, storeOp.getContext())));
      }
    } else {
      errs() << "Invalid user of memref: " << (*user) << '\n';
      return failure();
    }
  }

  return success();
}

static LogicalResult liftCaller(FuncOp callee, CallOp caller,
                                MemRefLiftInfo &info) {
  MLIRContext *ctx = callee.getContext();
  // Find the positions of the info.indicies in the callee.
  // Simple O(N^2) algo.
  SmallVector<unsigned> positions;
  for (auto index : info.indices)
    for (auto arg : enumerate(callee.getArguments()))
      if (index == arg.value())
        positions.push_back(arg.index());

  OpBuilder b(ctx);
  SmallVector<OpFoldResult> offsets, sizes, strides;

  // Get offsets.
  for (auto pos : positions) {
    Value offset = caller.getOperand(pos);
    LLVM_DEBUG(dbgs() << "Found offset " << offset << " for pos #" << pos
                      << '\n');
    offsets.push_back(offset);
  }
  for (unsigned i = info.indices.size(); i < info.type.getRank(); ++i)
    offsets.push_back(b.getIndexAttr(0));

  // Get sizes.
  for (unsigned i = 0; i < info.indices.size(); ++i)
    sizes.push_back(b.getIndexAttr(1));
  for (unsigned i = info.indices.size(); i < info.type.getRank(); ++i)
    sizes.push_back(b.getIndexAttr(info.type.getShape()[i]));

  // Get strides
  for (unsigned i = 0; i < info.type.getRank(); ++i)
    strides.push_back(b.getIndexAttr(1));

  // Create subview.
  b.setInsertionPoint(caller);

  /// NOTE: This subview type has a layout affine map, which is essential for
  /// getting memory access correct when lowering down to LLVM. However, it
  /// cannot work with the use-bare-ptr-memref-conv option in
  /// -convert-std-to-llvm. So we need to cast out the layout attribution when
  /// passing that among PEs.
  MemRefType subviewType = memref::SubViewOp::inferRankReducedResultType(
                               info.type.getRank() - info.indices.size(),
                               info.type, offsets, sizes, strides)
                               .cast<MemRefType>();
  auto subview = b.create<memref::SubViewOp>(caller.getLoc(), subviewType,
                                             caller.getOperand(info.index),
                                             offsets, sizes, strides);
  // Cast from its type to liftedType.
  auto cast =
      b.create<memref::CastOp>(caller.getLoc(), subview, info.getLiftedType());

  // Replace the original memref.
  caller.setOperand(info.index, cast);

  return success();
}

static LogicalResult process(FuncOp callee,
                             llvm::SetVector<Operation *> &callers,
                             MemRefLiftInfo &info) {

  // Use the subview for all the callers.
  if (failed(liftCallee(callee, info)))
    return failure();
  for (Operation *caller : callers)
    if (failed(liftCaller(callee, cast<CallOp>(caller), info)))
      return failure();

  return success();
}

static LogicalResult liftMemRefSubview(FuncOp callee,
                                       llvm::SetVector<Operation *> &callers) {
  // Get all the memref from the function arguments.
  llvm::MapVector<Value, MemRefLiftInfo> memrefs;
  for (auto arg : enumerate(callee.getArguments()))
    if (arg.value().getType().isa<MemRefType>())
      memrefs.insert({arg.value(), MemRefLiftInfo(arg.value(), arg.index())});

  // The common indices are ready to be lifted.
  resolveOffset(memrefs);

  for (auto &it : memrefs) {
    MemRefLiftInfo &info = it.second;

    if (info.indices.empty())
      continue;

    LLVM_DEBUG(dbgs() << "Indices size: " << info.indices.size() << '\n');

    if (failed(process(callee, callers, info)))
      return failure();
  }

  return success();
}

static void
findMemRefToAlias(const llvm::MapVector<Value, SmallVector<MemRefAccess>> &mas,
                  llvm::MapVector<Value, unsigned> &toAlias) {
  for (auto &it : mas) {
    Value memref = it.first;
    auto &accesses = it.second;
    unsigned numDimsToAlias = 0;

    for (auto &ma : accesses) {
      AffineValueMap avm;
      ma.getAccessMap(&avm);
      AffineMap am = avm.getAffineMap();

      unsigned offset = 0;
      for (AffineExpr expr : am.getResults()) {
        if (!expr.isa<AffineSymbolExpr>())
          break;
        ++offset;
      }

      numDimsToAlias = std::max(offset, numDimsToAlias);
    }

    if (numDimsToAlias) {
      toAlias[memref] = numDimsToAlias;
      LLVM_DEBUG(dbgs() << "We can alias " << memref << " to dim "
                        << numDimsToAlias << "\n");
    }
  }
}

struct AffineExprValue {
  AffineExpr expr;
  Value value;

  AffineExprValue(AffineExpr expr, Value value) : expr(expr), value(value) {}

  bool operator==(const AffineExprValue &rhs) const {
    return rhs.expr == expr && rhs.value == value;
  }

  int64_t offset() const {
    if (expr.isa<AffineSymbolExpr>())
      return 0;
    auto bin = expr.dyn_cast<AffineBinaryOpExpr>();
    if (bin.getKind() == AffineExprKind::Add &&
        bin.getLHS().isa<AffineSymbolExpr>() &&
        bin.getRHS().isa<AffineConstantExpr>())
      return bin.getRHS().cast<AffineConstantExpr>().getValue();
    if (bin.getKind() == AffineExprKind::FloorDiv &&
        bin.getLHS().isa<AffineSymbolExpr>() &&
        bin.getRHS().isa<AffineConstantExpr>())
      return 0;
    assert(false);
  }
};

using AffineExprValues = llvm::SmallVector<AffineExprValue>;
using AccessGroup = std::pair<AffineExprValues, SmallVector<MemRefAccess>>;

static void dump(const AffineExprValues &aevs) {
  dbgs() << "AffineExprValues(\n";
  for (const auto &aev : aevs)
    dbgs() << "\tExpr=" << aev.expr << ", Value=" << aev.value << '\n';
  dbgs() << ")\n";
}

static AffineExpr getSymbol(const AffineExpr &e) {
  if (e.isa<AffineSymbolExpr>())
    return e;
  auto bin = e.dyn_cast<AffineBinaryOpExpr>();
  if (bin.getKind() == AffineExprKind::Add &&
      bin.getLHS().isa<AffineSymbolExpr>() &&
      bin.getRHS().isa<AffineConstantExpr>())
    return bin.getLHS();
  if (bin.getKind() == AffineExprKind::FloorDiv &&
      bin.getLHS().isa<AffineSymbolExpr>() &&
      bin.getRHS().isa<AffineConstantExpr>())
    return bin.getLHS();
  return nullptr;
}

static void
buildAccessGroups(Value memref, FuncOp callee,
                  const llvm::SetVector<Operation *> &callers,
                  SmallVectorImpl<AccessGroup> &groups,
                  const llvm::MapVector<Value, SmallVector<MemRefAccess>> &mas,
                  const llvm::MapVector<Value, unsigned> &toAlias) {

  auto getAffineExprValues = [&](const AffineValueMap &avm,
                                 AffineExprValues &aevs) -> bool {
    aevs.clear();
    AffineMap am = avm.getAffineMap();
    for (unsigned i = 0; i < toAlias.lookup(memref); ++i) {
      AffineExpr expr = am.getResult(i);
      AffineExpr symbol = getSymbol(expr);
      if (!symbol)
        return false;
      Value value = getOperandByAffineExpr(avm, symbol);

      aevs.emplace_back(expr, value);
    }
    return true;
  };

  // group callers.
  // each alias is distinguished by the affine exprs and the operands.
  // each dim is a pair of AffineExpr and an affine symbol.
  for (MemRefAccess &ma : mas.lookup(memref)) {
    AffineValueMap avm;
    ma.getAccessMap(&avm);
    AffineMap am = avm.getAffineMap();

    AffineExprValues aevs;
    if (!getAffineExprValues(avm, aevs))
      continue;

    auto it = find_if(groups, [&](const AccessGroup &group) {
      return (group.first == aevs);
    });
    if (it == groups.end()) // no key
      groups.push_back({aevs, SmallVector<MemRefAccess>{ma}});
    else // has the key
      it->second.push_back(ma);
  }

  LLVM_DEBUG({
    dbgs() << "Access groups:\n";
    for (AccessGroup &ag : groups) {
      dbgs() << " - Key:\n";
      dump(ag.first);

      dbgs() << " - Accesses:\n";
      for (auto &ma : ag.second) {
        ma.opInst->dump();
        dbgs() << '\n';
      }
    }
  });
}

static LogicalResult
aliasSingleMemRef(Value memref, FuncOp callee,
                  llvm::SetVector<Operation *> &callers,
                  const llvm::MapVector<Value, SmallVector<MemRefAccess>> &mas,
                  const llvm::MapVector<Value, unsigned> &toAlias) {
  if (!(toAlias.count(memref) && toAlias.lookup(memref) > 0))
    return success();
  if (mas.lookup(memref).size() <= 1) // just one access, no need to alias.
    return success();

  LLVM_DEBUG(dbgs() << " --- Aliasing memref: " << memref << '\n');

  SmallVector<AccessGroup> ags;
  buildAccessGroups(memref, callee, callers, ags, mas, toAlias);

  auto needToAlias = [&](const AffineExprValues &aevs) {
    for (const AffineExprValue &aev : aevs)
      if (!aev.expr.isa<AffineSymbolExpr>())
        return true;
    return false;
  };

  auto findArgument = [&](FuncOp f, Value arg, unsigned &pos) -> bool {
    for (auto p : enumerate(f.getArguments()))
      if (p.value() == arg) {
        pos = p.index();
        return true;
      }
    return false;
  };

  auto findOperand = [&](Operation *op, Value arg, unsigned &pos) -> bool {
    for (auto p : enumerate(op->getOperands()))
      if (p.value() == arg) {
        pos = p.index();
        return true;
      }
    return false;
  };

  OpBuilder b(callee.getContext());
  Block *entry = &callee.getBlocks().front();
  for (AccessGroup &ag : ags) {
    const AffineExprValues &aevs = ag.first;
    SmallVector<MemRefAccess> &accesses = ag.second;

    if (!needToAlias(aevs)) {
      LLVM_DEBUG({
        dbgs() << "No need to alias the key -\n";
        dump(aevs);
      });
      continue;
    }

    Value memref = accesses.front().memref;

    unsigned id = 0; // current index
    Value newMemRef = entry->addArgument(memref.getType());
    for (const auto &aev : aevs) {
      // Add new arguments to the caller and change the access operation.
      Value newIndex = entry->addArgument(aev.value.getType());

      for (auto &ma : accesses) {
        AffineValueMap avm;
        ma.getAccessMap(&avm);
        AffineMap am = avm.getAffineMap();
        SmallVector<AffineExpr> results{am.getResults().begin(),
                                        am.getResults().end()};
        results[id] = getSymbol(results[id]);

        AffineMap newAm = AffineMap::get(am.getNumDims(), am.getNumSymbols(),
                                         results, callee.getContext());

        unsigned pos;
        assert(findOperand(ma.opInst, aev.value, pos));
        if (auto loadOp = dyn_cast<mlir::AffineLoadOp>(ma.opInst)) {
          loadOp.setMemRef(newMemRef);
          loadOp.setOperand(pos, newIndex);
          loadOp->setAttr(loadOp.getMapAttrName(), AffineMapAttr::get(newAm));

        } else if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(ma.opInst)) {
          storeOp.setMemRef(newMemRef);
          storeOp.setOperand(pos, newIndex);
          storeOp->setAttr(storeOp.getMapAttrName(), AffineMapAttr::get(newAm));
        }
      }

      for (Operation *caller : callers) {
        unsigned pos;
        assert(findArgument(callee, memref, pos));
        Value memrefOperand = caller->getOperand(pos);
        assert(findArgument(callee, aev.value, pos));

        b.setInsertionPoint(caller);

        MemRefType ty = memrefOperand.getType().dyn_cast<MemRefType>();
        assert(ty);

        Value indexOperand;
        if (aev.offset() < 0) {
          indexOperand = b.create<mlir::AffineMaxOp>(
              caller->getLoc(),
              AffineMap::get(0, 1, {aev.expr, b.getAffineConstantExpr(0)},
                             caller->getContext()),
              ValueRange(caller->getOperand(pos)));
        } else if (aev.offset() > 0) {
          indexOperand = b.create<mlir::AffineMinOp>(
              caller->getLoc(),
              AffineMap::get(
                  0, 1,
                  {aev.expr, b.getAffineConstantExpr(ty.getShape()[id] - 1)},
                  caller->getContext()),
              ValueRange(caller->getOperand(pos)));
        } else if (aev.expr.isa<mlir::AffineSymbolExpr>()) {
          indexOperand = caller->getOperand(pos);
        } else {
          indexOperand = b.create<mlir::AffineApplyOp>(
              caller->getLoc(), AffineMap::get(0, 1, aev.expr),
              ValueRange(caller->getOperand(pos)));
        }

        SmallVector<Value> newOperands;
        if (id == 0)
          newOperands.push_back(memrefOperand);
        newOperands.push_back(indexOperand);

        caller->insertOperands(caller->getNumOperands(), newOperands);
      }

      ++id;
    }
  }

  // TODO: make sure there is no returned value.
  callee.setType(b.getFunctionType(entry->getArgumentTypes(), TypeRange()));

  LLVM_DEBUG(dbgs() << "New callee:\n" << callee << '\n');

  return success();
}

static LogicalResult aliasMemRef(FuncOp callee,
                                 llvm::SetVector<Operation *> &callers) {
  LLVM_DEBUG(dbgs() << "Trying to alias memref for function: \n"
                    << callee << '\n');
  llvm::MapVector<Value, SmallVector<MemRefAccess>> mas;

  callee.walk([&](Operation *op) {
    if (isa<mlir::AffineLoadOp, mlir::AffineStoreOp>(op)) {
      MemRefAccess access(op);
      mas[access.memref].push_back(access);
    }
  });

  llvm::MapVector<Value, unsigned> toAlias;
  findMemRefToAlias(mas, toAlias);

  for (auto &it : mas)
    if (failed(aliasSingleMemRef(it.first, callee, callers, mas, toAlias)))
      return failure();

  return success();
}

static LogicalResult flattenPartitionDims(ModuleOp m) {
  LLVM_DEBUG(dbgs() << "Flattening ..." << m << '\n');
  using OpPair = std::pair<Operation *, Operation *>;
  OpBuilder b(m.getContext());
  llvm::MapVector<Value, SmallVector<OpPair>> mps;

  m.walk([&](Operation *op) {
    auto subview = dyn_cast<memref::SubViewOp>(op);
    if (!subview)
      return;

    Value source = subview.source();
    MemRefType srcTy = source.getType().cast<MemRefType>();
    LLVM_DEBUG(dbgs() << "Found subview " << subview << '\n');

    for (Operation *user : subview->getUsers()) {
      auto castOp = dyn_cast<memref::CastOp>(user);
      if (!castOp)
        return;

      MemRefType dstTy = castOp.getType().cast<MemRefType>();
      /// TODO: check if the rest of the rank are the same for src and dst.
      if (srcTy.getRank() - dstTy.getRank() >= 2)
        mps[source].push_back({subview, castOp});
    }
  });

  for (auto &it : mps) {
    Value memref = it.first;
    auto &pairs = it.second;

    assert(memref.isa<BlockArgument>());
    MemRefType newSrcTy;

    for (auto p : pairs) {
      auto subviewOp = cast<memref::SubViewOp>(p.first);
      auto castOp = cast<memref::CastOp>(p.second);
      LLVM_DEBUG(dbgs() << "Working on subview: " << subviewOp << '\n');
      Value source = subviewOp.source();
      Value dest = castOp.getResult();
      MemRefType srcTy = source.getType().cast<MemRefType>();
      MemRefType dstTy = dest.getType().cast<MemRefType>();

      unsigned ranks = srcTy.getRank() - dstTy.getRank();
      SmallVector<int64_t> shape;
      for (unsigned i = 0; i < ranks; ++i)
        shape.push_back(srcTy.getShape()[i]);
      // [32, 64] -> [64, 32]
      std::reverse(shape.begin(), shape.end());

      // [64, 32] -> [1, 64]
      SmallVector<int64_t> offset;
      offset.push_back(1);
      for (unsigned i = 1; i < ranks; ++i)
        offset.push_back(offset.back() * shape[i - 1]);
      LLVM_DEBUG({
        dbgs() << "Offset: {";
        interleaveComma(offset, dbgs());
        dbgs() << "}\n";
      });

      unsigned prod = 1;
      for (unsigned i = 0; i < ranks; ++i)
        prod *= shape[i];

      SmallVector<int64_t> newShape;
      newShape.push_back(prod);
      newShape.append(SmallVector<int64_t>(dstTy.getShape().begin(),
                                           dstTy.getShape().end()));
      newSrcTy = MemRefType::Builder(newShape, srcTy.getElementType());

      // [1, 64] -> [64, 1]
      std::reverse(offset.begin(), offset.end());
      LLVM_DEBUG({
        dbgs() << "Offset: {";
        interleaveComma(offset, dbgs());
        dbgs() << "}\n";
      });
      AffineExpr index = b.getAffineConstantExpr(0);
      for (unsigned i = 0; i < ranks; ++i)
        index =
            index + b.getAffineDimExpr(i) * b.getAffineConstantExpr(offset[i]);
      LLVM_DEBUG(dbgs() << "Index expr: " << index << '\n');

      b.setInsertionPoint(subviewOp);

      SmallVector<Value> operands;
      bool isMin = false;
      bool isMinOrMax = false;
      // HACK:
      for (Value value : subviewOp.offsets().take_front(ranks)) {
        if (auto minOp = value.getDefiningOp<mlir::AffineMinOp>()) {
          assert(!isMinOrMax);
          isMin = true;
          isMinOrMax = true;
          operands.push_back(minOp.getOperand(0));
        } else if (auto maxOp = value.getDefiningOp<mlir::AffineMaxOp>()) {
          assert(!isMinOrMax);
          isMin = false;
          isMinOrMax = true;
          operands.push_back(maxOp.getOperand(0));
        } else {
          operands.push_back(value);
        }
      }

      LLVM_DEBUG(dbgs() << "isMinOrMax = " << isMinOrMax << " isMin = " << isMin
                        << '\n');

      Value newIndex;
      if (!isMinOrMax) {
        newIndex = b.create<mlir::AffineApplyOp>(
            subviewOp.getLoc(), AffineMap::get(ranks, 0, index), operands);
      } else {
        if (isMin) {
          newIndex = b.create<mlir::AffineMinOp>(
              subviewOp.getLoc(),
              AffineMap::get(ranks, 0,
                             {index, b.getAffineConstantExpr(prod) - 1},
                             b.getContext()),
              operands);
        } else {
          newIndex = b.create<mlir::AffineMaxOp>(
              subviewOp.getLoc(),
              AffineMap::get(ranks, 0, {index, b.getAffineConstantExpr(0)},
                             b.getContext()),
              operands);
        }
      }

      SmallVector<OpFoldResult> offsets, sizes, strides;
      offsets.push_back(newIndex);
      sizes.push_back(b.getIndexAttr(1));
      strides.push_back(b.getIndexAttr(1));
      for (unsigned i = 0; i < dstTy.getRank(); ++i) {
        offsets.push_back(b.getIndexAttr(0));
        strides.push_back(b.getIndexAttr(1));
        sizes.push_back(b.getIndexAttr(dstTy.getShape()[i]));
      }

      LLVM_DEBUG({
        dbgs() << "Offsets: {";
        interleaveComma(offsets, dbgs());
        dbgs() << "}\n";
      });
      LLVM_DEBUG({
        dbgs() << "Sizes: {";
        interleaveComma(sizes, dbgs());
        dbgs() << "}\n";
      });
      LLVM_DEBUG({
        dbgs() << "Strides: {";
        interleaveComma(strides, dbgs());
        dbgs() << "}\n";
      });

      MemRefType newSubviewType =
          memref::SubViewOp::inferRankReducedResultType(
              dstTy.getRank(), newSrcTy, offsets, sizes, strides)
              .cast<MemRefType>();
      LLVM_DEBUG(dbgs() << "New src type: " << newSrcTy << "\n"
                        << newSubviewType << '\n');
      auto newSubview = b.create<memref::SubViewOp>(
          subviewOp.getLoc(), newSubviewType, memref, offsets, sizes, strides);
      // Cast from its type to liftedType.
      auto newCast =
          b.create<memref::CastOp>(subviewOp.getLoc(), newSubview, dstTy);

      castOp.replaceAllUsesWith(newCast.getResult());

      castOp.erase();
      subviewOp.erase();
    }

    BlockArgument arg = memref.dyn_cast<BlockArgument>();
    assert(arg);

    auto findArgument = [&](FuncOp f, Value arg, unsigned &pos) -> bool {
      for (auto p : enumerate(f.getArguments()))
        if (p.value() == arg) {
          pos = p.index();
          return true;
        }
      return false;
    };

    FuncOp f = cast<FuncOp>(arg.getOwner()->getParentOp());

    unsigned pos;
    assert(findArgument(f, memref, pos));

    arg.setType(newSrcTy);
    f.setType(b.getFunctionType(arg.getOwner()->getArgumentTypes(),
                                f.getType().getResults()));
  }

  return success();
}

namespace {
struct LiftMemRefSubviewPass
    : public ::phism::LiftMemRefSubviewBase<LiftMemRefSubviewPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Map from a function to all its callees in the top.
    llvm::DenseMap<FuncOp, llvm::SetVector<Operation *>> callerMap;
    m.walk([&](CallOp caller) {
      if (!caller->hasAttr("phism.pe"))
        return;
      FuncOp callee =
          dyn_cast_or_null<FuncOp>(m.lookupSymbol(caller.getCallee()));
      if (!callee)
        return;
      callerMap[callee].insert(caller);
    });

    for (auto &it : callerMap)
      if (failed(aliasMemRef(it.first, it.second)))
        return signalPassFailure();

    LLVM_DEBUG(dbgs() << "Module after aliased:\n" << m << '\n');

    // For each entry, first rewrite the callee, then update all the
    // callers.
    for (auto &it : callerMap)
      if (failed(liftMemRefSubview(it.first, it.second))) {
        errs() << "Failed to lift memref subview for function: \n"
               << it.first << '\n';
        return signalPassFailure();
      }

    if (flatten)
      if (failed(flattenPartitionDims(m)))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createLiftMemRefSubviewPass() {
  return std::make_unique<LiftMemRefSubviewPass>();
}
