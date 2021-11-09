//===- Utils.cc -------------------------------------------------*- C++ -*-===//
// LLVM transform utilities.
//===----------------------------------------------------------------------===//

#include "phism/llvm/Transforms/Utils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

static cl::opt<std::string>
    XlnTop("xlntop", cl::desc("Specify the top function for Xilinx HLS."),
           cl::value_desc("topname"));
static cl::opt<std::string>
    XlnNames("xlnnames", cl::desc("Specify the top function param names."),
             cl::value_desc("paramname"));
static cl::opt<std::string> XlnTBTclNames(
    "xlntbtclnames",
    cl::desc(
        "Specify the file name of the tcl script for test bench generation."),
    cl::value_desc("tbname"));
static cl::opt<std::string> XlnTBDummyNames(
    "xlntbdummynames",
    cl::desc("Specify the file name of the C dummy for test bench generation."),
    cl::value_desc("dummyname"));
static cl::opt<std::string>
    XlnLLVMIn("xlnllvm", cl::desc("Specify the LLVM source for the design."),
              cl::value_desc("llvm input"));
static cl::opt<bool> XlnArrayPartitionEnabled(
    "xln-ap-enabled", cl::desc("Whether array partition has been enabled"));
static cl::opt<bool> XlnArrayPartitionFlattened(
    "xln-ap-flattened", cl::desc("Whether array partition has been flattened"));

namespace phism {
namespace llvm {

bool isPointerToArray(Type *type) {
  return type->isPointerTy() && type->getPointerElementType()->isArrayTy();
}

std::string getXlnTop() { return XlnTop; }
std::string getXlnNames() { return XlnNames; }
std::string getXlnTBTclNames() { return XlnTBTclNames; }
std::string getXlnTBDummyNames() { return XlnTBDummyNames; }
std::string getXlnLLVMIn() { return XlnLLVMIn; }
bool getXlnArrayPartitionEnabled() { return XlnArrayPartitionEnabled; }
bool getXlnArrayPartitionFlattened() { return XlnArrayPartitionFlattened; }

} // namespace llvm
} // namespace phism
