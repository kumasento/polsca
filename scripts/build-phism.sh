#!/usr/bin/env bash

# ----------- Install Phism (experimental) --------------
# It is not guaranteed to work seamlessly on any machine. Do check the
# exact commands it runs if anything wrong happens, and post the error
# messages here: https://github.com/kumasento/phism/issues

set -o errexit
set -o pipefail
set -o nounset

echo ""
echo ">>> Install Phism "
echo ""

# ------------------------- Environment --------------------------

# TARGET can be local or ci. ci build won't use the Xilinx environment
TARGET="${1:-"local"}"

# The absolute path to the directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PHISM_DIR="${DIR}/.."
LLVM_DIR="${PHISM_DIR}/llvm"

if [ "${TARGET}" == "local" ]; then
  "${DIR}/check-vitis.sh" || { echo "Xilinx Vitis check failed."; exit 1; }
fi


# ------------------------- CMake Configure ---------------------

cd "${PHISM_DIR}"
mkdir -p build
cd build
CC=gcc CXX=g++ cmake .. \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR="${LLVM_DIR}/build/lib/cmake/mlir/" \
  -DLLVM_DIR="${LLVM_DIR}/build/lib/cmake/llvm/" \
  -DLLVM_EXTERNAL_LIT="${LLVM_DIR}/build/bin/llvm-lit" 

# ------------------------- Build and test ---------------------

cmake --build . --target check-phism
