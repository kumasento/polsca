# Phism: Polyhedral High-Level Synthesis in MLIR

[![Build and Test](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml)

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

[pb-flow](scripts/pb-flow) provides a CLI utility to test Phism with Polybench examples. You can grab a rough idea about the whole Phism pipeline over there.

More usage info will be added later.
