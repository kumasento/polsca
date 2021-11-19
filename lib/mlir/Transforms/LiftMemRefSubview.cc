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

#define DEBUG_TYPE "array-partition"

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

      // figure out the new offset and pop the back.
      for (unsigned i = 0, e = std::min(info.indices.size(), indices.size());
           i < e; ++i) {
        if (info.indices[i] != indices[i]) {
          // New offset determined.
          info.indices.pop_back_n(info.indices.size() - i);
          break;
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

    if (failed(process(callee, callers, info)))
      return failure();
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

    // For each entry, first rewrite the callee, then update all the
    // callers.
    for (auto &it : callerMap)
      if (failed(liftMemRefSubview(it.first, it.second))) {
        errs() << "Failed to lift memref subview for function: \n"
               << it.first << '\n';

        return signalPassFailure();
      }
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createLiftMemRefSubviewPass() {
  return std::make_unique<LiftMemRefSubviewPass>();
}
