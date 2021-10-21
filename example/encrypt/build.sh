#!/usr/bin/env bash

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

mlir-clang "${DIR}/encrypt.c" \
  -I "${LLVM_DIR}/build/lib/clang/14.0.0/include" \
  -S \
  --function=encrypt \
  --raise-scf-to-affine |\
  polymer-opt
