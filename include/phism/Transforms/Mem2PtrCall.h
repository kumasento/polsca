//===- Mem2PtrCall.h - mem2ptr-call transformation ------------------------===//
//
// This file declares the -mem2ptr-call transformation pass, which replace the
// original function body by a new call that takes in parameters of the pointers
// to the memrefs.
//
//===----------------------------------------------------------------------===//

#ifndef PHISM_TRANSFORMS_MEM2PTRCALL_H
#define PHISM_TRANSFORMS_MEM2PTRCALL_H

namespace phism {
/// Register the -mem2ptr-call pass that
void registerMem2PtrCallPass();
} // namespace phism

#endif
