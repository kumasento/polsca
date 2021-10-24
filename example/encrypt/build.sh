#!/usr/bin/env bash

# TODO: this is just a draft script.

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ROOT_DIR="${DIR}/../.."
POLYGEIST_DIR="${ROOT_DIR}/polygeist"
LLVM_DIR="${POLYGEIST_DIR}/llvm-project"
POLYMER_DIR="${ROOT_DIR}/polymer"

export PATH="${POLYGEIST_DIR}/build/mlir-clang:${PATH}"
export PATH="${POLYMER_DIR}/build/bin:${PATH}"
export PATH="${LLVM_DIR}/build/bin:${PATH}"
export PATH="${ROOT_DIR}/build/bin:${PATH}"

BUILD_DIR="${ROOT_DIR}/build/example/encrypt"
mkdir -p "${BUILD_DIR}"

gcc "${DIR}/encrypt.c" -o "${BUILD_DIR}/encrypt.bin"
"${BUILD_DIR}/encrypt.bin" > "${BUILD_DIR}/result.golden"

# Test polygeist
mlir-clang "${DIR}/encrypt.c" -I "${LLVM_DIR}/build/lib/clang/14.0.0/include" -S -O0 -memref-fullrank -raise-scf-to-affine > "${BUILD_DIR}/encrypt.mlir"
mlir-opt "${BUILD_DIR}/encrypt.mlir" -lower-affine -convert-scf-to-std -convert-memref-to-llvm -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -convert-arith-to-llvm -reconcile-unrealized-casts |\
mlir-translate -mlir-to-llvmir |\
opt -O3 |\
lli > "${BUILD_DIR}/result.polygeist"
diff "${BUILD_DIR}/result.golden" "${BUILD_DIR}/result.polygeist"

# Test polymer
mlir-clang "${DIR}/encrypt.c" -I "${LLVM_DIR}/build/lib/clang/14.0.0/include" -S -O0 -memref-fullrank -raise-scf-to-affine |\
  polymer-opt -annotate-scop="functions=encrypt" -fold-scf-if -reg2mem -extract-scop-stmt -pluto-opt |\
  mlir-opt -lower-affine -convert-scf-to-std -convert-memref-to-llvm -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -convert-arith-to-llvm -reconcile-unrealized-casts |\
  mlir-translate -mlir-to-llvmir |\
  opt -O3 |\
  lli > "${BUILD_DIR}/result.polymer"
diff "${BUILD_DIR}/result.golden" "${BUILD_DIR}/result.polymer"

# Test phism
mlir-clang "${DIR}/encrypt.c" -I "${LLVM_DIR}/build/lib/clang/14.0.0/include" -S -O0 -memref-fullrank -raise-scf-to-affine |\
  phism-opt -extract-top-func="name=encrypt keepall=1" |\
  polymer-opt -annotate-scop="functions=encrypt" -fold-scf-if -reg2mem -extract-scop-stmt -pluto-opt |\
  phism-opt -loop-transforms -scop-stmt-inline |\
  mlir-opt -lower-affine -convert-scf-to-std -convert-memref-to-llvm -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -convert-arith-to-llvm -reconcile-unrealized-casts |\
  mlir-translate -mlir-to-llvmir |\
  opt -O3 |\
  lli > "${BUILD_DIR}/result.phism"
diff "${BUILD_DIR}/result.golden" "${BUILD_DIR}/result.phism"

# Lower to LLVM
mlir-clang "${DIR}/encrypt.c" -I "${LLVM_DIR}/build/lib/clang/14.0.0/include" -S -O0 -memref-fullrank -raise-scf-to-affine |\
  phism-opt -extract-top-func="name=encrypt keepall=1" |\
  polymer-opt -annotate-scop="functions=encrypt" -fold-scf-if -reg2mem -extract-scop-stmt -pluto-opt |\
  phism-opt -loop-transforms -scop-stmt-inline |\
  mlir-opt -lower-affine -convert-scf-to-std -convert-memref-to-llvm -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -convert-arith-to-llvm -reconcile-unrealized-casts |\
  mlir-translate -mlir-to-llvmir |\
  opt -S -enable-new-pm=0 -load="${ROOT_DIR}/build/lib/VhlsLLVMRewriter.so" -strip-debug -mem2arr -instcombine -xlntop='encrypt' -xlnnames='Sbox,statemt' -xlnmath -xlnname -xlnanno -xlnunroll -xlnram2p -strip-attr
