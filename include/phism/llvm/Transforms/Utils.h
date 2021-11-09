//===- Utils.h --------------------------------------------------*- C++ -*-===//
// LLVM transform utilities.
//===----------------------------------------------------------------------===//

#include <string>

namespace llvm {
class Type;
}

namespace phism {
namespace llvm {

bool isPointerToArray(::llvm::Type *);

std::string getXlnTop();
std::string getXlnNames();
std::string getXlnTBTclNames();
std::string getXlnTBDummyNames();
std::string getXlnLLVMIn();
bool getXlnArrayPartitionEnabled();
bool getXlnArrayPartitionFlattened();

} // namespace llvm
} // namespace phism
