//===- ArrayPartition.cc - array partition -------------------------- C++-===//
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

/// If the corresponding result is from an identity input, get the corresponding
/// value.
static Value getIdentityMapOperand(const AffineValueMap &avm, unsigned index) {
  if (index >= avm.getNumResults())
    return nullptr;

  AffineExpr result = avm.getAffineMap().getResult(index);
  if (auto expr = result.dyn_cast<AffineDimExpr>())
    return avm.getOperand(expr.getPosition());
  if (auto expr = result.dyn_cast<AffineSymbolExpr>())
    return avm.getOperand(expr.getPosition() + avm.getNumDims());
  return nullptr;
}

/// Unified interface to collect memref from operations.
static Value getMemRef(Operation *op) {
  if (isa<mlir::AffineReadOpInterface, memref::LoadOp>(op))
    return op->getOperand(0);
  else if (isa<mlir::AffineWriteOpInterface, memref::StoreOp>(op))
    return op->getOperand(1);
  return nullptr;
}

static Value getAddr(Operation *op, unsigned index) {
  if (isa<mlir::AffineReadOpInterface, memref::LoadOp>(op))
    return op->getOperand(index + 1);
  else if (isa<mlir::AffineWriteOpInterface, memref::StoreOp>(op))
    return op->getOperand(index + 2);
  return nullptr;
}

static AffineValueMap getAffineValueMap(Operation *op) {
  AffineValueMap avm;
  if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
    MemRefAccess access(op);
    access.getAccessMap(&avm);
  }
  return avm;
}

namespace {
using MemRefMapping = llvm::DenseMap<Value, Value>;

/// TODO: may need to redesign this class.
struct Partition {
  enum Kind { NONE, BLOCK } kind;
  int64_t block;

  /// Initialize block partition.
  Partition() : kind(NONE) {}
  Partition(int64_t block) : kind(BLOCK), block(block) {}

  void dump() const;
};

void Partition::dump() const {
  errs() << "Partition(";
  if (kind == Kind::BLOCK)
    errs() << "BLOCK, " << block << ")";
  else
    errs() << "NONE)";
}

struct Access {
  Operation *op;
  bool isAffine;
  Value memref;
  AffineValueMap avm;
  SmallVector<Partition, 4> parts;

  explicit Access(Operation *op)
      : op(op),
        isAffine(
            isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)),
        memref(getMemRef(op)), avm(getAffineValueMap(op)) {}

  void dump() const;

  /// Find valid array partition patterns from the op.
  void initPartitions();

  /// Check different partition patterns.

  /// Block partition can be satisfied if the address index is a loop indvar,
  /// and the bounds are (doesn't matter if the input is symbol or dim):
  ///    #map0 = affine_map<()[s0] -> (s0 * T)>
  ///    #map1 = affine_map<()[s0] -> (s0 * T + T)>
  /// Constant bounds will be filtered.
  LogicalResult createBlockPartition(unsigned index, Partition &part) const;
};

void Access::dump() const {
  errs() << "\t* op:       " << (*op) << '\n';
  errs() << "\t* isAffine: " << (isAffine) << '\n';
  errs() << "\t* memref:   " << (memref) << '\n';
  if (isAffine)
    errs() << "\t* affMap:   " << (avm.getAffineMap()) << '\n';

  for (unsigned i = 0; i < parts.size(); ++i) {
    errs() << "\t* Part[" << i << "]:  ";
    parts[i].dump();
    errs() << "\n";
  }
}

void Access::initPartitions() {
  MemRefType ty = memref.getType().dyn_cast<MemRefType>();
  unsigned rank = ty.getRank();

  parts.resize(rank);
  for (unsigned ind = 0; ind < rank; ++ind) {
    if (succeeded(createBlockPartition(ind, parts[ind])))
      continue;
  }
}

LogicalResult Access::createBlockPartition(unsigned index,
                                           Partition &part) const {
  if (!isAffine)
    return failure();

  MemRefAccess access(op);
  AffineValueMap avm;
  access.getAccessMap(&avm);

  Value addr = getIdentityMapOperand(avm, index);
  if (!addr)
    return failure();

  auto forOp = getForInductionVarOwner(addr);
  AffineMap lbMap = filterExtraConstantResults(forOp.getLowerBoundMap());
  AffineMap ubMap = filterExtraConstantResults(forOp.getLowerBoundMap());

  // Match map structure.
  if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1 ||
      lbMap.getNumDims() != ubMap.getNumDims() ||
      lbMap.getNumSymbols() != ubMap.getNumSymbols() ||
      lbMap.getNumInputs() != 1 || ubMap.getNumInputs() != 1)
    return failure();
  // Match the index value.
  if (forOp.getLowerBoundOperands()[0] != forOp.getUpperBoundOperands()[0])
    return failure();

  /// Match T (d0 | s0) + T. If failed, return -1.
  auto matchMulAdd = [&](AffineExpr expr) -> int64_t {
    AffineBinaryOpExpr bin = expr.dyn_cast<AffineBinaryOpExpr>();
    if (!bin)
      return -1;

    // a x + b
    AffineExpr x;
    AffineConstantExpr a, b;
    // Try to parse the add operation.
    if (bin.getKind() == AffineExprKind::Add) {
      if (bin.getLHS().isa<AffineConstantExpr>()) {
        // Left constant
        b = bin.getLHS().cast<AffineConstantExpr>();
        bin = bin.getRHS().dyn_cast<AffineBinaryOpExpr>();
      } else {
        // Right constant
        b = bin.getRHS().cast<AffineConstantExpr>();
        bin = bin.getLHS().dyn_cast<AffineBinaryOpExpr>();
      }
    }

    // Parse the multiplier.
    if (bin.getKind() == AffineExprKind::Mul) {
      a = bin.getLHS().dyn_cast<AffineConstantExpr>();
      x = bin.getRHS();
      if (!a) {
        a = bin.getRHS().dyn_cast<AffineConstantExpr>();
        x = bin.getLHS();
      }
    }

    // Post-check
    if (!x || !a)
      return -1;
    if (b && (a.getValue() != b.getValue()))
      return -1;
    return a.getValue();
  };

  auto getPositionAndValue = [&](AffineExpr x, AffineExpr a, unsigned &pos,
                                 int64_t &value) {
    if (auto expr = x.dyn_cast<AffineDimExpr>())
      pos = expr.getPosition();
    if (auto expr = x.dyn_cast<AffineSymbolExpr>())
      pos = expr.getPosition();
  };

  int64_t T = matchMulAdd(lbMap.getResult(0));
  if (T <= 0 || T != matchMulAdd(ubMap.getResult(0)))
    return failure();

  // Set the result.
  part = Partition(T);
  return success();
}

struct MemRefPartition {
  Value memref;
  SmallVector<Partition, 4> parts;

  MemRefPartition(Value memref)
      : memref(memref), parts(memref.getType().cast<MemRefType>().getRank()) {}

  void dump() const;

  AffineMap getAffineMap() const;

  MemRefType getPartitionedType() const;
};

void MemRefPartition::dump() const {
  for (unsigned i = 0; i < parts.size(); ++i) {
    errs() << "\t* Part[" << i << "]: ";
    parts[i].dump();
    errs() << '\n';
  }

  errs() << "\n\t* AffineMap:\n";
  getAffineMap().dump();
  errs() << '\n';
}

AffineMap MemRefPartition::getAffineMap() const {
  MLIRContext *ctx = memref.getContext();
  unsigned rank = parts.size();

  // TODO: assuming block partition across all dimensions.
  SmallVector<AffineExpr> indices(rank * 2);
  for (unsigned ind = 0; ind < rank; ++ind) {
    const Partition &part = parts[ind];
    assert(part.kind == Partition::Kind::BLOCK);

    AffineExpr dim = getAffineDimExpr(ind, ctx);
    AffineExpr block = getAffineConstantExpr(part.block, ctx);
    indices[ind] = dim.floorDiv(block);
    indices[ind + rank] = dim % block;
  }

  return AffineMap::get(rank, 0, indices, ctx);
}

MemRefType MemRefPartition::getPartitionedType() const {
  MLIRContext *ctx = memref.getContext();

  MemRefType ty = memref.getType().cast<MemRefType>();
  unsigned rank = ty.getRank();

  const auto &origShape = ty.getShape();
  SmallVector<int64_t> shape(rank * 2);
  assert(all_of(parts, [](const Partition &part) {
    return part.kind == Partition::Kind::BLOCK;
  }));

  for (unsigned i = 0; i < rank; ++i) {
    shape[i] = std::ceil((double)origShape[i] / parts[i].block);
    shape[i + rank] = parts[i].block;
  }

  // TODO: could be dangerous if the AffineMaps from the original type is not
  // empty.
  return MemRefType::get(shape, ty.getElementType(), ty.getAffineMaps(),
                         ty.getMemorySpace());
}

struct MemRefToPartition {
  // map from memref to each address partition.
  llvm::DenseMap<Value, MemRefPartition> map_;

  void dump() const;
};

void MemRefToPartition::dump() const {
  errs() << "\nReconciled partitions:\n";
  for (const auto &it : map_) {
    const Value &memref = it.first;
    const auto &partition = it.second;

    errs() << "- MemRef: " << memref << "\n";
    partition.dump();
  }
}

struct MemRefToAccess {
  // accesses.
  llvm::DenseMap<Value, SmallVector<Access, 4>> map_;

  /// Go through every operation within the top, as well as all the nested
  /// callers. Don't consider at this stage that some memrefs might be the same
  /// object from the top.
  void build(FuncOp top, ModuleOp m);

  void dump() const;

  /// Every memref, if not exists in MemRefMapping as the source, it is a
  /// top-level memref. We want to make sure that every memref in the `mta` is
  /// top-level.
  void aggregate(const MemRefMapping &mrm);

  /// Go through every access and initialize its partitions.
  void initAccessPartitions();

  /// Try to find a unified partition for each memref.
  LogicalResult reconcilePartitions(MemRefToPartition &) const;
};

void MemRefToAccess::build(FuncOp top, ModuleOp m) {
  top.walk([&](Operation *op) {
    if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface,
            memref::LoadOp, memref::StoreOp>(op)) {
      Value memref = getMemRef(op);
      assert(memref);
      map_[memref].emplace_back(op);
    } else if (auto caller = dyn_cast<CallOp>(op)) {
      FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
      build(callee, m);
    }
  });
}

void MemRefToAccess::dump() const {
  for (const auto &it : map_) {
    const Value &memref = it.first;
    const auto &accesses = it.second;

    errs() << "\n + MemRef:" << memref << "\n\n";
    for (unsigned i = 0; i < accesses.size(); ++i) {
      errs() << " -> Access #" << i << " :\n";
      accesses[i].dump();
    }
  }
}

void MemRefToAccess::aggregate(const MemRefMapping &mrm) {
  bool changed = false;

  // Make sure every item in mrm exists as key in mta.
  for (auto &it : mrm) {
    if (!map_.count(it.first))
      map_[it.first] = {};
    if (!map_.count(it.second))
      map_[it.second] = {};
  }

  // Iterative algorithm. Each iteration, tries to propagate the accesses of
  // one memref one step forward.
  do {
    changed = false;

    for (auto &it : map_) {
      const Value &srcMem = it.first;
      if (!it.second.empty() && mrm.count(srcMem)) {
        const Value &dstMem = mrm.lookup(srcMem);
        LLVM_DEBUG(dbgs() << " * From " << srcMem << " to " << dstMem << '\n');
        map_[dstMem].append(map_[srcMem]);
        map_[srcMem].clear();

        changed = true;
      }
    }
  } while (changed);

  // Clear empty accesses.
  for (auto it = map_.begin(); it != map_.end();) {
    if (it->second.empty())
      map_.erase(it++);
    else
      it++;
  }
}

void MemRefToAccess::initAccessPartitions() {
  for (auto &it : map_)
    for (auto &access : it.second)
      access.initPartitions();
}

LogicalResult
MemRefToAccess::reconcilePartitions(MemRefToPartition &mtp) const {
  for (auto &it : map_) {
    const Value &memref = it.first;
    const auto &accesses = it.second;

    MemRefType ty = memref.getType().dyn_cast<MemRefType>();
    unsigned rank = ty.getRank();

    // TODO: this is the simplest algorithm.
    // Initially all partitions are initialized to NONE.
    auto ret = mtp.map_.insert({memref, MemRefPartition(memref)});
    auto &parts = ret.first->second.parts;
    parts.resize(rank);

    for (const auto &access : accesses) {
      for (unsigned ind = 0; ind < rank; ++ind) {
        const Partition &curr = parts[ind];
        const Partition &part = access.parts[ind];

        // Reconcile rules.
        // TODO: any better ways of formulating these?
        if (curr.kind == Partition::Kind::BLOCK) {
          // Ignore NONE partitions.
          if (part.kind == Partition::Kind::NONE)
            continue;
          // Won't reconcile if there are different block sizes.
          // TODO: possibly choose the larger one? No harm though.
          else if (part.kind == Partition::Kind::BLOCK) {
            if (part.block != curr.block)
              return failure();
          } else {
            llvm_unreachable("Unrecognized kind.");
          }
        } else if (curr.kind == Partition::Kind::NONE) {
          // Ignore NONE partitions.
          if (part.kind == Partition::Kind::NONE)
            continue;
          else if (part.kind == Partition::Kind::BLOCK) {
            parts[ind] = part;
          } else {
            llvm_unreachable("Unrecognized kind.");
          }
        } else {

          llvm_unreachable("Unrecognized kind.");
        }
      }
    }

    // TODO: for those addresses undecided, turn into block by its own size.
  }

  return success();
}

} // namespace

/// Map memref from the callee block argument to the caller operands.
static void buildMemRefMapping(FuncOp f, MemRefMapping &mrm, ModuleOp m) {
  f.walk([&](CallOp caller) {
    FuncOp callee = dyn_cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
    if (callee) {
      for (unsigned i = 0; i < callee.getNumArguments(); ++i) {
        Value arg = callee.getArgument(i);
        if (arg.getType().isa<MemRefType>())
          mrm[arg] = caller.getOperand(i);
      }

      // Recursion.
      buildMemRefMapping(callee, mrm, m);
    }
  });
}

/// Propagate the change of memref type all the way down.
/// memref in the argument list is the corresponding value in the current scope.
static void propagate(FuncOp f, Value memref, const MemRefPartition &mp,
                      MemRefType ty, ModuleOp m) {
  MLIRContext *ctx = f.getContext();

  // TODO: memref can be something defined within f.
  assert(memref.isa<BlockArgument>());
  assert(memref.cast<BlockArgument>().getOwner()->getParentOp() == f);

  unsigned argInd = find(f.getArguments(), memref) - f.args_begin();
  LLVM_DEBUG(dbgs() << "MemRef is at: " << argInd << '\n');

  SmallVector<Type> argTypes(f.getArgumentTypes());

  argTypes[argInd] = ty;
  FunctionType fty = FunctionType::get(ctx, argTypes, f.getType().getResults());
  f.setType(fty);
  LLVM_DEBUG(dbgs() << "New function type: " << fty << '\n');

  Block &entryBlock = *f.getBlocks().begin();
  entryBlock.getArgument(argInd).setType(ty);
  LLVM_DEBUG(dbgs() << "Replaced entry block argument type.\n");

  // Replace the access.
  AffineMap affMap = mp.getAffineMap();
  for (Operation *user : memref.getUsers()) {
    if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(user)) {
      MemRefAccess access(user);
      AffineValueMap avm;
      access.getAccessMap(&avm);

      AffineMap newAffMap = affMap.compose(avm.getAffineMap());
      if (auto loadOp = dyn_cast<mlir::AffineLoadOp>(user))
        loadOp->setAttr(loadOp.getMapAttrName(), AffineMapAttr::get(newAffMap));
      else if (auto storeOp = dyn_cast<mlir::AffineStoreOp>(user))
        storeOp->setAttr(loadOp.getMapAttrName(),
                         AffineMapAttr::get(newAffMap));
    }
  }

  f.walk([&](CallOp caller) {
    FuncOp callee = cast<FuncOp>(m.lookupSymbol(caller.getCallee()));
    for (unsigned i = 0; i < caller.getNumOperands(); ++i)
      if (caller.getOperand(i) == memref) {
        LLVM_DEBUG(dbgs() << "--> To propagate caller: " << caller
                          << " at index: " << i << '\n');
        propagate(callee, callee.getArgument(i), mp, ty, m);
      }
  });
}

static LogicalResult partitionMemRef(FuncOp top, const MemRefPartition &mp,
                                     ModuleOp m, OpBuilder &b) {
  LLVM_DEBUG(dbgs() << "---> Partitioning MemRef: " << mp.memref << "\n");
  LLVM_DEBUG(dbgs() << "Top function - \n" << top << '\n');

  // Replace the source memref by the new type.
  MemRefType newTy = mp.getPartitionedType();
  LLVM_DEBUG(dbgs() << "New memref type: " << newTy << '\n');

  propagate(top, mp.memref, mp, newTy, m);

  return success();
}

namespace {
struct ArrayPartitionPass
    : public phism::ArrayPartitionBase<ArrayPartitionPass> {

  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // Get the top function.
    FuncOp top = findPhismTop(m);
    if (!top) {
      m.emitRemark() << "No top function found for array partition. Have you "
                        "forgot to annotate {scop.pe} to callers?\n";
      return;
    }

    // Before transformation, keep all the existing functions into a set so that
    // they won't be recycled later.
    SmallPtrSet<FuncOp, 4> keep;
    getFunctionsToKeep(m, top, keep);

    // Build the mapping from MemRef to all its access operations, including
    // affine and non-affine.
    MemRefToAccess mta;
    mta.build(top, m);
    LLVM_DEBUG({
      dbgs() << "\n================== Got mta:\n";
      mta.dump();
      dbgs() << '\n';
    });

    // Map memref values from the callee to the caller.
    // Ultimately, we want to get a mapping that map from memref to their
    // top-level correspondent.
    // Top-level means: block arguments in phism.top, or those allocated within
    // each scope.
    MemRefMapping mrm;
    buildMemRefMapping(top, mrm, m);

    mta.aggregate(mrm);
    LLVM_DEBUG({
      dbgs() << "\n================== After aggregation:\n";
      mta.dump();
      dbgs() << '\n';
    });

    mta.initAccessPartitions();
    LLVM_DEBUG({
      dbgs() << "\n================== Partitions initialized:\n";
      mta.dump();
      dbgs() << '\n';
    });

    // Try to reconcile the partition results.
    MemRefToPartition mtp;
    if (failed(mta.reconcilePartitions(mtp)))
      return signalPassFailure();

    LLVM_DEBUG({
      dbgs() << "\n================== Reconciled:\n";
      mtp.dump();
      dbgs() << '\n';
    });

    for (auto &it : mtp.map_)
      if (failed(partitionMemRef(top, it.second, m, b)))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createArrayPartitionPass() {
  return std::make_unique<ArrayPartitionPass>();
}
