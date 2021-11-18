//===- PhismTransforms.cc ---------------------------------------*- C++ -*-===//
//
// This file implements the registration interfaces for all Phism passes.
//
//===----------------------------------------------------------------------===//

#include "phism/mlir/Transforms/PhismTransforms.h"
#include "phism/mlir/Transforms/Passes.h"

namespace phism {

void registerAllPhismPasses() {
  registerLoopTransformPasses();
  registerExtractTopFuncPass();
  // registerDependenceAnalysisPasses();
  registerFoldIfPasses();
  registerLoopBoundHoistingPass();
  registerEliminateAffineLoadStorePass();
  registerSplitNonAffinePass();
  registerArrayPartitionPass();
  registerSimpleArrayPartitionPass();
  registerSimplifyPartitionAccessPass();
  registerStripExceptTopPass();
  registerAffineLoopUnswitchingPass();
  registerAnnotatePointLoopPass();
  registerOutlineProcessElementPass();
  registerRewritePloopIndvarPass();
}

} // namespace phism
