//===- PhismTransforms.cc ---------------------------------------*- C++ -*-===//
//
// This file implements the registration interfaces for all Phism passes.
//
//===----------------------------------------------------------------------===//

#include "phism/mlir/Transforms/PhismTransforms.h"

namespace phism {

void registerAllPhismPasses() { registerExtractTopFuncPass(); }

} // namespace phism