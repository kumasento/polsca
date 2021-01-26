//===- Mem2PtrCall.cc - mem2ptr-call transformation -----------------------===//
//
// This file implements the -mem2ptr-call transformation pass.
//
//===----------------------------------------------------------------------===//

#include "phism/Transforms/Mem2PtrCall.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"

using namespace llvm;
using namespace mlir;

static void
getMemRefStructDefOps(LLVM::LLVMFuncOp f,
                      llvm::SmallVectorImpl<Operation *> &memDefOps) {
  Block &entryBlock = f.body().front();
  for (Operation &op : entryBlock.getOperations()) {
    if (!isa<LLVM::UndefOp, LLVM::InsertValueOp>(op))
      break;
    memDefOps.push_back(&op);
  }

  // Sanity checks on size.
  assert(memDefOps.size() % 6 == 0 && "The total number of memref def ops "
                                      "should be 6 * N (number of memrefs).");
  // Sanity checks on op types.
  for (unsigned off = 0; off < memDefOps.size(); off += 6) {
    assert(isa<LLVM::UndefOp>(memDefOps[off]) &&
           "The first op of each def group should be a LLVM::UndefOp.");
    assert(llvm::all_of(
               llvm::make_range(memDefOps.begin() + off + 1,
                                memDefOps.begin() + off + 6),
               [&](Operation *op) { return isa<LLVM::InsertValueOp>(op); }) &&
           "The rest of each def group should be of type LLVM::InsertValueOp.");
  }

  // TODO: check the use-def chain.
}

static LLVM::LLVMFuncOp
createCallee(Block *block, llvm::ArrayRef<Operation *> memDefOps,
             llvm::SmallVectorImpl<mlir::Value> &calleeArgs, ModuleOp m,
             OpBuilder &b) {
  // Find the values that should be treated as block arguments based on: is of
  // type BlockArgument; or defined out of the current scope.
  llvm::SetVector<mlir::Value> argSet;
  block->walk([&](Operation *op) {
    for (mlir::Value operand : op->getOperands())
      if (operand.isa<mlir::BlockArgument>() ||
          operand.getDefiningOp()->getBlock() != block) {
        // If any argument is used by "extractvalue" internally, we should move
        // that value out of this current block. The actual way of handling is:
        // set the values defined by these "extractvalue" as arguments, and when
        // jumping back to the original region, we create new "extractvalue"
        // ops.
        //
        // Example:
        //
        // ^bb0:
        // ^bb1:
        //    %1 = llvm.extractvalue %0[1]
        //
        // After transformation, we have
        //
        // ^bb0:
        //   %1 = llvm.extractvalue %0[1]
        //   call @f(%1)

        if (llvm::any_of(operand.getUsers(), [&](Operation *user) {
              return isa<LLVM::ExtractValueOp>(user);
            })) {
          assert(operand.hasOneUse() &&
                 "The operand should not be used by `extractvalue`, otherwise, "
                 "it should have a single use.");

          // TODO: detect if that operand is indeed one from memDefOps.
          // Instead of inserting the operand, we get its single user's single
          // result.
          argSet.insert(operand.getUsers().begin()->getResult(0));
        } else {
          argSet.insert(operand);
        }
      }
  });

  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (mlir::Value arg : argSet) {
    calleeArgs.push_back(arg);
    argTypes.push_back(arg.getType());
  }

  Operation *termOp = block->getTerminator();
  assert(termOp->getNumOperands() <= 1 &&
         "Number of operands of a LLVMFuncOp terminator should be <= 1");
  mlir::Type retType = termOp->getNumOperands()
                           ? termOp->getOperand(0).getType()
                           : b.getNoneType();

  LLVM::LLVMFuncOp parent =
      block->getParent()->getParentOfType<LLVM::LLVMFuncOp>();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(parent);
  LLVM::LLVMFuncOp callee = b.create<LLVM::LLVMFuncOp>(
      parent->getLoc(), std::string("_") + std::string(parent.getName()),
      LLVM::LLVMFunctionType::get(retType, argTypes));
  Block *entryBlock = callee.addEntryBlock();

  BlockAndValueMapping mapping;
  mapping.map(argSet.getArrayRef(), entryBlock->getArguments());

  b.setInsertionPointToStart(entryBlock);
  block->walk([&](Operation *op) {
    if (isa<LLVM::ExtractValueOp>(op))
      return;
    b.clone(*op, mapping);
  });

  return callee;
}

static void applyMem2ptrCall(Operation *funOp, ModuleOp m, OpBuilder &b) {
  LLVM::LLVMFuncOp f = dyn_cast<LLVM::LLVMFuncOp>(funOp);
  assert(f != nullptr && "`funOp` provided should be of type LLVMFuncOp.");
  assert(f.getBlocks().size() == 1 && "`funOp` should only have one Block.");

  Block &entryBlock = f.body().front();
  OpBuilder::InsertionGuard guard(b);

  // Get memref "definition" ops.
  llvm::SmallVector<Operation *, 8> memDefOps;
  getMemRefStructDefOps(f, memDefOps);

  // Split the entryBlock into two by the last "insertvalue" op.
  Block *newEntryBlock = entryBlock.splitBlock(memDefOps.back()->getNextNode());
  assert(newEntryBlock->use_empty() &&
         "The split entry block should have no use.");

  // Create the callee around the newly split block.
  llvm::SmallVector<mlir::Value, 8> calleeArgs;
  LLVM::LLVMFuncOp callee =
      createCallee(newEntryBlock, memDefOps, calleeArgs, m, b);

  // Create a caller to the new callee.
  b.setInsertionPointToEnd(&entryBlock);

  // If any callee arguments are defined by `extractvalue`, and the defining ops
  // are in the newEntryBlock, we should clone them before calling the newly
  // created callee, and replace these args by the results of these new
  // `extractvalue` functions. Reasons for doing so can be found in
  // `createCallee`.
  BlockAndValueMapping mapping;
  for (mlir::Value arg : calleeArgs) {
    if (LLVM::ExtractValueOp defOp =
            arg.getDefiningOp<LLVM::ExtractValueOp>()) {
      if (defOp->getBlock() != newEntryBlock)
        continue;

      Operation *newDefOp = b.clone(*arg.getDefiningOp(), mapping);
      mapping.map(arg, newDefOp->getResult(0));
    }
  }

  // Create the caller and the final return operand.
  llvm::SmallVector<mlir::Value, 8> callerArgs;
  for (mlir::Value arg : calleeArgs)
    callerArgs.push_back(mapping.lookup(arg));
  LLVM::CallOp caller =
      b.create<LLVM::CallOp>(callee.getLoc(), callee, callerArgs);
  if (caller->getNumResults() == 0)
    b.create<LLVM::ReturnOp>(caller.getLoc(), llvm::None);
  else
    b.create<LLVM::ReturnOp>(caller.getLoc(), caller->getResult(0));

  newEntryBlock->erase();
}

namespace {
class Mem2PtrCallPass
    : public mlir::PassWrapper<Mem2PtrCallPass, OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // Only interested in `llvm.func`. Pre-cache these Ops since the module
    // will be later modified (is also the reason why we don't use
    // FunctionPass).
    llvm::SmallVector<Operation *, 8> worklist;
    m.walk([&](Operation *op) {
      if (isa<LLVM::LLVMFuncOp>(op))
        worklist.push_back(op);
    });

    for (Operation *op : worklist)
      applyMem2ptrCall(op, m, b);
  }
};
} // namespace

namespace phism {
void registerMem2PtrCallPass() {
  PassRegistration<Mem2PtrCallPass>(
      "mem2ptr-call",
      "Create inner functions of ptrs if memrefs are in the arg list.");
}
} // namespace phism
