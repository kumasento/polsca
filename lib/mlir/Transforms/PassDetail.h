//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//

#ifndef PHISM_MLIR_TRANSFORMS_PASSDETAIL_H_
#define PHISM_MLIR_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "phism/mlir/Transforms/Passes.h"

namespace phism {
#define GEN_PASS_CLASSES
#include "phism/mlir/Transforms/Passes.h.inc"
} // namespace phism

#endif // TRANSFORMS_PASSDETAIL_H_
