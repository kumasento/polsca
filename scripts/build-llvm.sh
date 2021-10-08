#!/usr/bin/env bash
# This script installs the llvm shipped together with Phism.

set -o errexit
set -o pipefail
set -o nounset

echo ""
echo ">>> Install LLVM for Phism"
echo ""

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# The absolute path to the directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure llvm submodule is up-to-date.
git submodule sync
git submodule update --init --recursive

# Go to the llvm directory and carry out installation.
LLVM_DIR="${DIR}/../polygeist/llvm-project"

cd "${LLVM_DIR}"
mkdir -p build
cd build

# Configure CMake
if [ ! -f "CMakeCache.txt" ]; then
  cmake ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_LINKER=lld \
    -G "${CMAKE_GENERATOR}"
fi 
 
# Run building
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
  ninja check-mlir
else 
  make -j "$(nproc)"
fi
