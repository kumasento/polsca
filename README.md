# Phism: Polyhedral High-Level Synthesis in MLIR

![Build and Test](https://github.com/kumasento/phism/workflows/Build%20and%20Test/badge.svg)

## Setup

Install prerequisites for [MLIR/LLVM](https://mlir.llvm.org/getting_started/) and [Pluto](https://github.com/kumasento/pluto/blob/master/README.md).

Specifically, you need:

* (LLVM) `cmake` >= 3.13.4.
* (LLVM) Valid compiler tool-chain that supports C++ 14
* (Pluto) Automatic build tools (for Pluto), including `autoconf`, `automake`, and `libtool`.
* (Pluto) Pre-built LLVM-9 tools (`clang-9` and `FileCheck-9`) and their header files are needed.
* (Pluto) `libgmp` that is required by isl.
* (Pluto) `flex` and `bison` for `clan` that Pluto depends on.
* (Pluto) `texinfo` used to generate some docs.

Here is a one-liner on Ubuntu 20.04:

```shell
sudo apt-get install -y build-essential libtool autoconf pkg-config flex bison libgmp-dev clang-9 libclang-9-dev texinfo
```

On Ubuntu you may need to specify the default versions of these tools:

```shell
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 100
sudo update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 100
```

## Install

Download this repository:

```shell
git clone https://github.com/kumasento/phism
```

Update the submodules:

```shell
git submodule update 
```

### LLVM

To install LLVM:

```shell
# At the top-level directory 
mkdir llvm/build
cd llvm/build
cmake ../llvm \
  -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -G Ninja
cmake --build . -- -j$(nproc)
cmake --build . --target check-mlir
```

### Polymer

TBA

### This project

```shell
# At the top-level directory 
mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_EXTERNAL_LIT=${PWD}/../llvm/build/bin/llvm-lit \
cmake --build . 

# Unit tests
cmake --build . --target check-phism
```
