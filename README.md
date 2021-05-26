# Phism: Polyhedral High-Level Synthesis in MLIR

[![Self-hosted Build and Test](https://github.com/kumasento/phism/actions/workflows/self-hosted-build-and-test.yml/badge.svg)](https://github.com/kumasento/phism/actions/workflows/self-hosted-build-and-test.yml)

## What is Phism?

Phism is an HLS tool: it transforms C programs to hardware designs.

Phism leverages [MLIR](https://mlir.llvm.org) and enjoys the _progressive lowering_ idea to build the full compilation pipeline. The ability to apply a full range of compilation transformations in an organized, layered way is what Phism stands out from other tools.

Phism optimises hardware design generation through polyhedral modelling, a powerful technique that especially good at transforming statically scheduled nested loops for better parallelism and locality.

## How to build?

### Prerequisites 

Please find how to setup the prerequisites [here](docs/PREREQUISITES.md).

### Build LLVM

After that, the first thing you need to do is building the [Polygeist](wsmoses/Polygeist) submodule (with name `llvm`). Make sure you have it cloned:

```sh
git submodule update --init --update
```

You may see many submodules being synced -- don't worry, they are simply required by Pluto, the polyhedral optimizer that Phism uses.

To build Polygeist, Just run the following script. It should take care of everything you need.

```sh
./scripts/build-llvm.sh
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
