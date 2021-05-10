# Phism: Polyhedral High-Level Synthesis in MLIR

[![Build and Test](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml)

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
