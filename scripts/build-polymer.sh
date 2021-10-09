#!/usr/bin/env bash
# This script installs the Polymer repository.

set -o errexit
set -o pipefail
set -o nounset

echo ""
echo ">>> Install Polymer for Phism"
echo ""

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# The absolute path to the directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure Polymer submodule is up-to-date.
git submodule sync
git submodule update --init --recursive

LLVM_DIR="${DIR}/../polygeist/llvm-project"
# Go to the polymer directory and carry out installation.
POLYMER_DIR="${DIR}/../polymer"

cd "${POLYMER_DIR}"
mkdir -p build
cd build

ls -al .

echo ""
echo ">>> Compiler info:"
echo ""

clang --version
clang++ --version

# Configure CMake
if [ ! -f "CMakeCache.txt" ]; then
  cmake -G "${CMAKE_GENERATOR}" \
    .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_LINKER=lld \
    -DMLIR_DIR="${LLVM_DIR}/build/lib/cmake/mlir" \
    -DLLVM_DIR="${LLVM_DIR}/build/lib/cmake/llvm" \
    -DLLVM_EXTERNAL_LIT="${LLVM_DIR}/build/bin/llvm-lit" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
fi 
 
# Run building
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
  ninja check-polymer
else 
  make -j "$(nproc)"
fi
