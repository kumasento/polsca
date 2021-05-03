//===- PhismTransforms.h ----------------------------------------*- C++ -*-===//
//
// This file declares all the registration interfaces for Phism passes.
//
//===----------------------------------------------------------------------===//

#ifndef PHISM_MLIR_TRANSFORMS_PHISMTRANSFORMS_H
#define PHISM_MLIR_TRANSFORMS_PHISMTRANSFORMS_H

namespace phism {
void registerExtractTopFuncPass();
void registerAllPhismPasses();
} // namespace phism

#endif
