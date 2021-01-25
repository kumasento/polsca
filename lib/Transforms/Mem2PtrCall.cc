//===- Mem2PtrCall.cc - mem2ptr-call transformation -----------------------===//
//
// This file implements the -mem2ptr-call transformation pass.
//
//===----------------------------------------------------------------------===//

#include "phism/Transforms/Mem2PtrCall.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace llvm;
using namespace mlir;

namespace {
class Mem2PtrCallPass
    : public mlir::PassWrapper<Mem2PtrCallPass, OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {}
};
} // namespace

namespace phism {
void registerMem2PtrCallPass() {
  PassRegistration<Mem2PtrCallPass>(
      "mem2ptr-call",
      "Create inner functions of ptrs if memrefs are in the arg list.");
}
} // namespace phism
