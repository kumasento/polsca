//===- Utils.cc - Utility functions ------------------ C++-===//

#include "phism/mlir/Transforms/Utils.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace llvm;
using namespace phism;

static bool hasPeCaller(FuncOp f) {
  bool ret = false;
  f.walk([&](CallOp caller) {
    if (caller->hasAttr("scop.pe"))
      ret = true;
  });
  return ret;
}

namespace phism {

FuncOp getTopFunction(ModuleOp m) {
  FuncOp top = nullptr;
  m.walk([&](FuncOp f) {
    if (hasPeCaller(f)) {
      assert(!top && "There should be only one top function.");
      top = f;
    }
  });
  return top;
}

} // namespace phism
