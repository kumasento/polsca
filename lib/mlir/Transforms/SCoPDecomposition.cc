/// SCoPDecomposition.cc

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

#define DEBUG_TYPE "scop-decomp"

using namespace llvm;
using namespace mlir;
using namespace phism;

using MemRefAccesses = llvm::DenseMap<Value, SmallVector<MemRefAccess>>;

/// TODO: any side product to keep?
static bool hasDependence(mlir::AffineForOp src, mlir::AffineForOp dst,
                          int64_t depth) {

  LLVM_DEBUG(dbgs() << "Checking dependencies between: \n"
                    << src << "\n and:\n"
                    << dst << "\n");

  // --- Step 1: get all memrefs and accesses
  MemRefAccesses srcMas, dstMas;
  auto getMemRefAccesses = [](mlir::AffineForOp loop, MemRefAccesses &mas) {
    mas.clear();
    /// TODO: assuming there are only affine load/store.
    loop.walk([&](Operation *op) {
      if (isa<mlir::AffineLoadOp, mlir::AffineStoreOp>(op)) {
        MemRefAccess access(op);
        mas[access.memref].push_back(access);
      }
    });
  };

  getMemRefAccesses(src, srcMas);
  getMemRefAccesses(dst, dstMas);

  //  --- Step 2: find common memrefs.
  SmallVector<Value> commonMemRefs;
  for (auto &it : srcMas)
    if (dstMas.count(it.first))
      commonMemRefs.push_back(it.first);

  LLVM_DEBUG({
    dbgs() << "Common MemRefs:\n";
    for (Value memref : commonMemRefs)
      dbgs() << " * " << memref << "\n";
  });

  // --- Step 3: check dependencies among each memref access pair.
  /// TODO: we assume there is no alias.
  for (Value memref : commonMemRefs) {
    for (const MemRefAccess &srcAccess : srcMas[memref]) {
      for (const MemRefAccess &dstAccess : dstMas[memref]) {
        LLVM_DEBUG(dbgs() << "Checking dependencies between src:\n"
                          << (*srcAccess.opInst) << "\n  and dst:\n"
                          << (*dstAccess.opInst) << "\n  at depth: " << depth
                          << '\n');

        FlatAffineValueConstraints depCst;
        SmallVector<DependenceComponent, 2> depComps;
        auto dep = checkMemrefAccessDependence(srcAccess, dstAccess, depth,
                                               &depCst, &depComps,
                                               /*allowRAR=*/false);
        if (dep.value == DependenceResult::HasDependence)
          return true;
      }
    }
  }

  return false;
}

// Get the minimum number of separation cuts.
static void getMinCuts(const SmallVectorImpl<llvm::SetVector<unsigned>> &deps,
                       SmallVectorImpl<int64_t> &seps) {
  seps.clear();

  // A DP algo.
  auto N = deps.size();
  if (N < 2)
    return;
  if (N == 2) { // The simplest case.
    if (deps[0].count(1))
      seps.push_back(0); // cut at 0.
    return;
  }

  // minCuts[i][len] gives the (minimum) cuts required to make the interval has
  // zero interval.
  SmallVector<SmallVector<SmallVector<int64_t>>> minCuts(
      N, SmallVector<SmallVector<int64_t>>(N + 1, SmallVector<int64_t>()));

  // initialize
  for (unsigned i = 0; i < N; ++i)
    minCuts[i][0] = minCuts[i][1] = {}; // no need to cut.
  for (unsigned i = 0; i + 1 < N; ++i)
    if (deps[i].count(i + 1))
      minCuts[i][2] = {i};

  for (unsigned len = 3; len <= N; ++len) {
    for (unsigned i = 0; i + len <= N; ++i) {
      // Check if there is any need to cut by seeing if there is any dependence
      // from any element within the interval that has destination within the
      // interval as well.
      // bool needToCut = false;
      // for (unsigned j = i; j < i + len; ++j)
      //   for (unsigned k : deps[j])
      //     if (k < i + len)
      //       needToCut = true;

      int64_t minVal = std::numeric_limits<int64_t>::max();
      int64_t pos = -1;       // There is no place to insert a new cut.
      bool needToCut = false; // do we need to cut for the best answer.

      // [i, j) and [j, i + len)
      for (unsigned j = i + 1; j < i + len; ++j) {
        LLVM_DEBUG(dbgs() << " -> Examining [" << i << ", " << j << ") [" << j
                          << ", " << i + len << ") ...\n");
        LLVM_DEBUG({
          dbgs() << " * minCuts[" << i << "][" << j - i << "] = {";
          interleaveComma(minCuts[i][j - i], dbgs());
          dbgs() << "}\n";
        });
        LLVM_DEBUG({
          dbgs() << " * minCuts[" << j << "][" << i + len - j << "] = {";
          interleaveComma(minCuts[j][i + len - j], dbgs());
          dbgs() << "}\n";
        });

        unsigned curVal =
            minCuts[i][j - i].size() + minCuts[j][i + len - j].size();
        LLVM_DEBUG(dbgs() << " * curVal = " << curVal << '\n');

        /// TODO: can improve with a segment tree or simply prefix sum to give a
        /// quicker query for the existence of element beyond a boundary.
        bool need = false;
        unsigned lastPos =
            minCuts[i][j - i].empty() ? i : minCuts[i][j - i].back() + 1;
        unsigned firstPos = minCuts[j][i + len - j].empty()
                                ? i + len
                                : minCuts[j][i + len - j].front() + 1;

        LLVM_DEBUG(dbgs() << " * Checking deps between lastPos = " << lastPos
                          << " and firstPos = " << firstPos << '\n');

        for (unsigned k = lastPos; k < j; ++k)
          for (unsigned d : deps[k])
            if (d >= j && d < firstPos)
              need = true;
        LLVM_DEBUG(dbgs() << " * need to cut = " << need << '\n');

        if (curVal + need < minVal) {
          minVal = curVal + needToCut;
          pos = j - 1;
          needToCut = need;
        }
      }

      LLVM_DEBUG(dbgs() << " = Pos determined: " << pos << '\n');

      minCuts[i][len].append(minCuts[i][pos - i + 1]);
      if (needToCut)
        minCuts[i][len].push_back(pos);
      minCuts[i][len].append(minCuts[pos + 1][len - (pos - i + 1)]);

      LLVM_DEBUG({
        dbgs() << " = minCuts[" << i << "][" << len << "] = {";
        interleaveComma(minCuts[i][len], dbgs());
        dbgs() << "}\n";
      });
    }

    seps = minCuts[0][N];
  }
}

static LogicalResult decompositeSCoP(Block *block, const int64_t depth,
                                     const int64_t maxLoopDepth, FuncOp f,
                                     unsigned &id) {
  if (maxLoopDepth != 0 && depth >= maxLoopDepth)
    return success();

  // Assume all the affine for loops are valid SCoPs.
  SmallVector<mlir::AffineForOp> loops;
  for (Operation &op : *block)
    if (auto loop = dyn_cast<mlir::AffineForOp>(&op))
      loops.push_back(loop);

  if (loops.empty()) // cannot decomposite for sure.
    return success();
  if (loops.size() == 1) // just go deeper, no chance of decompositing here.
    return decompositeSCoP(loops.front().getBody(), depth + 1, maxLoopDepth, f,
                           id);

  // Calculate all the dependencies among loops.
  SmallVector<llvm::SetVector<unsigned>> deps(loops.size());
  for (unsigned i = 0; i < loops.size(); ++i)
    for (unsigned j = i + 1; j < loops.size(); ++j)
      if (hasDependence(loops[i], loops[j], depth))
        deps[i].insert(j);

  LLVM_DEBUG({
    dbgs() << "Dependencies:\n";
    for (unsigned i = 0; i < loops.size(); ++i) {
      dbgs() << "From loop #" << i << " -> ";
      interleaveComma(deps[i], dbgs());
      dbgs() << '\n';
    }
  });

  // Find the minimum cuts. Each sep in seps gives the cut position, e.g., seps
  // = {0, 2} means cutting after the first and the third loops.
  SmallVector<int64_t> seps;
  getMinCuts(deps, seps);

  LLVM_DEBUG({
    dbgs() << "Separations: ";
    interleaveComma(seps, dbgs());
    dbgs() << '\n';
  });

  // There is no need to cut at this depth.
  if (seps.empty()) {
    for (auto loop : loops)
      if (failed(
              decompositeSCoP(loop.getBody(), depth + 1, maxLoopDepth, f, id)))
        return failure();
    return success();
  }

  OpBuilder b(f.getContext());

  // Exists seps, should separate.
  unsigned prev = 0;
  for (unsigned i = 0; i <= seps.size(); ++i) {
    unsigned curr = i < seps.size() ? seps[i] + 1 : loops.size();

    SmallVector<Operation *> subLoops{loops.begin() + prev,
                                      loops.begin() + curr};
    std::string name = std::string(f.getName()) + "__f" + std::to_string(id);
    auto p = outlineFunction(subLoops, name, f->getParentOfType<ModuleOp>());
    auto callee = dyn_cast_or_null<FuncOp>(p.first);
    auto caller = dyn_cast_or_null<CallOp>(p.second);
    assert(callee && caller);

    callee->setAttr("scop.affine", b.getUnitAttr());
    caller->setAttr("scop.affine", b.getUnitAttr());

    ++id;
    prev = curr;

    for (auto op : subLoops)
      op->erase();
  }

  f->setAttr("scop.ignore", b.getUnitAttr());

  return success();
}

static LogicalResult decompositeSCoP(FuncOp f, const int64_t maxLoopDepth) {
  unsigned id = 0;
  for (Block &block : f.getBlocks())
    if (failed(decompositeSCoP(&block, 0, maxLoopDepth, f, id)))
      return failure();
  return success();
}

namespace {
struct SCoPDecompositionPass
    : public ::phism::SCoPDecompositionBase<SCoPDecompositionPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    /// TODO: support other functions in general.
    FuncOp f = findPhismTop(m);
    if (!f)
      return;

    if (failed(decompositeSCoP(f, maxLoopDepth)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
phism::createSCoPDecompositionPass() {
  return std::make_unique<SCoPDecompositionPass>();
}
