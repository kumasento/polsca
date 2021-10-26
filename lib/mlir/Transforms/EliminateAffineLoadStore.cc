//===- EliminateAffineLoadStore.cc ------------------------------*- C++ -*-===//
//
// Implements a pass that eliminates redundant affine load store operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

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

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "eliminate-affine-load-store"

using namespace mlir;
using namespace llvm;
using namespace phism;

using AccessOpList = std::pair<MemRefAccess, SmallVector<Operation *>>;

/// Iterate every operations in the block. Put load and store operations into
/// the corresponding memref access buckets. If there is an operation that has
/// blocks, they will be a NULL separator in every bucket.
static void getAccessOps(Block *block,
                         SmallVectorImpl<AccessOpList> &accessOps) {
  for (Operation &op : *block) {
    if (!isa<mlir::AffineLoadOp, mlir::AffineStoreOp>(op)) {
      // If there appears to be a inner block, we currently assume it can
      // have side effects on all the memref accesses.
      // We insert an delimeter (NULL) to all the existing accessOps lists.
      if (op.getNumRegions() >= 1)
        for (auto &p : accessOps)
          p.second.push_back(nullptr);

      continue;
    }

    // Insert or create.
    MemRefAccess access(&op);
    auto it = find_if(accessOps,
                      [&](const AccessOpList &p) { return p.first == access; });
    if (it == accessOps.end())
      accessOps.push_back({access, SmallVector<Operation *>{&op}});
    else
      it->second.push_back(&op);
  }
}

/// For each block under the provided function, we find all the load
/// operations to each accessed address. If multiple loads to the same address
/// have no interleaved store operations (to the same address), we merge them
/// into one.
///
/// An address being accessed is considered the same if
static LogicalResult eliminateIdenticalAffineLoadOps(FuncOp f, OpBuilder &b) {

  std::function<void(Block *)> processBlock = [&](Block *block) {
    SmallVector<AccessOpList, 4> accessOps;
    getAccessOps(block, accessOps);

    LLVM_DEBUG({
      dbgs() << "Resolved memref accesses and their operations:\n";
      for (auto &p : accessOps) {
        p.first.memref.dump();
        for (Operation *op : p.second) {
          if (op)
            op->dump();
          else
            dbgs() << "NULL\n";
        }
        dbgs() << "\n";
      }
    });

    // Eliminate redundant load operations.
    for (auto &p : accessOps) {
      Operation *curr = nullptr;
      for (Operation *op : p.second) {
        if (isa_and_nonnull<mlir::AffineLoadOp>(op)) {
          // If there is no replace target, set curr.
          if (!curr)
            curr = op;
          else { // Carry out replacement.
            op->replaceAllUsesWith(curr);
            op->erase();
          }
        } else {
          // Reset the front load operation in each section.
          curr = nullptr;
        }
      }
    }

    // Process internal blocks.
    for (Operation &op : *block)
      for (Region &region : op.getRegions())
        for (Block &blk : region)
          processBlock(&blk);
  };

  for (Block &block : f.getBody())
    processBlock(&block);

  return success();
}

/// In cases that an affine.load follows an affine.store that operate on the
/// same address, we will try to eliminate them and replace the use of the load
/// by the values to be stored.
static LogicalResult eliminateLoadAfterStore(FuncOp f, OpBuilder &b) {
  std::function<void(Block *)> processBlock = [&](Block *block) {
    SmallVector<AccessOpList, 4> accessOps;
    getAccessOps(block, accessOps);

    for (auto &p : accessOps) {
      auto &ops = p.second;

      // Keeps the last-seen affine.store operation.
      Operation *prevStore = nullptr;
      // Keeps all the stores that should be erased, in visiting order.
      SmallVector<Operation *> storesToErase;

      for (auto op : enumerate(ops)) {
        if (isa_and_nonnull<mlir::AffineLoadOp>(op.value())) { // load
          if (prevStore) {
            op.value()->getResult(0).replaceAllUsesWith(
                prevStore->getOperand(0));
            op.value()->erase();
          }
        } else if (isa_and_nonnull<mlir::AffineStoreOp>(op.value())) { // store
          prevStore = op.value();
          storesToErase.push_back(op.value());
        } else { // block
          assert(!op.value());
          // Remove the last store operation from the stack.
          if (!storesToErase.empty())
            storesToErase.pop_back();
        }
      }

      // The last store should be kept.
      if (!storesToErase.empty())
        storesToErase.pop_back();

      for (Operation *op : storesToErase)
        op->erase();
    }

    // Process internal blocks.
    for (Operation &op : *block)
      for (Region &region : op.getRegions())
        for (Block &blk : region)
          processBlock(&blk);
  };

  for (Block &block : f.getBody())
    processBlock(&block);

  return success();
}

namespace {
struct EliminateAffineLoadStore
    : public EliminateAffineLoadStoreBase<EliminateAffineLoadStore> {

  void runOnFunction() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    if (failed(eliminateIdenticalAffineLoadOps(f, b)) ||
        (loadAfterStore && failed(eliminateLoadAfterStore(f, b)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
phism::createEliminateAffineLoadStorePass() {
  return std::make_unique<EliminateAffineLoadStore>();
}
