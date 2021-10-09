# Phism: Polyhedral High-Level Synthesis in MLIR

[![Build and Test](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/kumasento/phism)
![GitHub](https://img.shields.io/github/license/kumasento/phism)
![GitHub issues](https://img.shields.io/github/issues/kumasento/phism)
![GitHub pull requests](https://img.shields.io/github/issues-pr/kumasento/phism)


## What is Phism?

Phism is an HLS tool: it transforms C programs to hardware designs.

Phism leverages [MLIR](https://mlir.llvm.org) and enjoys the _progressive lowering_ idea to build the full compilation pipeline. The ability to apply a full range of compilation transformations in an organized, layered way is what Phism stands out from other tools.

Phism optimises hardware design generation through polyhedral modelling, a powerful technique that especially good at transforming statically scheduled nested loops for better parallelism and locality.

## How to build?

### Prerequisites 

Please find how to setup the prerequisites [here](docs/PREREQUISITES.md).

### Build LLVM

Phism uses [Polygeist](https://github.com/wsmoses/Polygeist) to process C/C++ code into MLIR. But before we built Polygeist, we need to build the LLVM package within Polygeist (`polygeist/llvm-project`). This LLVM package will be shared by Polygeist, Polymer (later), and Phism.

First of all, make sure you've initialized all the submodules.

```sh
git submodule update --init --recursive
```

You may see many submodules being synced -- don't worry, they are simply required by Pluto, the polyhedral optimizer that Phism uses.

To build LLVM, Just run the following script. It should take care of everything you need.

```sh
./scripts/build-llvm.sh
```

### Build Polygeist

It is another one-liner:

```sh
./script/build-polygeist.sh
```

### Build Polymer

[Polymer](https://github.com/kumasento/polymer) provides the functionality to interact MLIR code with polyhedral scheduler.

There is also a script for you - 

```sh
./script/build-polymer.sh
```

### Build Phism

Finally, you're all prepared to build Phism! Just type in the following commands:

```sh
./scripts/build-phism.sh
```

It should run the Phism regression test in the end. And if all the tests passed, hooray!

## Usage

### Using Docker

This [doc](docs/DOCKER.md) gives an introduction on how to run Phism with docker.

## Evaluation and benchmarking

### Polybench

Polybench is the major benchmark we look at. It contains 30 different examples covering programs that can be potentially optimised by polyhedral transformation. The benchmark suite is located at [example/polybench](example/polybench), and you can find our report [here](docs/POLYBENCH.md).
