#!/usr/bin/env bash
# This script installs the Polygeist repository.

set -o errexit
set -o pipefail
set -o nounset

echo ""
echo ">>> Install Polygeist for Phism"
echo ""

TARGET="${1:-"local"}"

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# The absolute path to the directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure Polygeist submodule is up-to-date.
git submodule sync

if [ ! -d "${DIR}/../polygeist/llvm-project" ]; then
  git submodule update --init --recursive
fi

# Go to the polygeist directory and carry out installation.
POLYGEIST_DIR="${DIR}/../polygeist"

cd "${POLYGEIST_DIR}"
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
    -DMLIR_DIR="${POLYGEIST_DIR}/llvm-project/build/lib/cmake/mlir" \
    -DCLANG_DIR="${POLYGEIST_DIR}/llvm-project/build/lib/cmake/clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
fi 
 
# Run building
if [ "${CMAKE_GENERATOR}" == "Ninja" ]; then
  ninja
  ninja check-mlir-clang
else 
  make -j "$(nproc)"
fi
