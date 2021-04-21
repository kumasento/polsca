# Phism: Polyhedral High-Level Synthesis in MLIR

[![Build and Test](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/kumasento/phism/actions/workflows/buildAndTest.yml)

## How to build?

### Prerequisites

#### OS

The operating system we target is Linux, specifically, we test Phism on Ubuntu 20.04 and CentOS 7.6.

#### Vitis

> Please ensure you follow these steps carefully. They are necessary for working seamlessly with the toolchain encapsulated within Vitis. You may hit unexpected linking issue if the environment is not properly set.

You need to have Vitis on your machine before installing Phism. The version we're currently using is 2020.2. For more information please visit [this page](https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html).

Once you have Vitis installed, you need to source its settings script:

```sh
# Suppose Vitis is installed under /opt
source /opt/Vitis/2020.2/settings64.sh
```

And you should be able to see the following output if the previous command is successful:

```sh
echo $XILINX_VITIS
# Should print /opt/Vitis/2020.2
echo $XILINX_HLS
# Should print /opt/Vitis_HLS/2020.2
```

After that, run the script [setup-vitis-hls-llvm.sh](scripts/setup-vitis-hls-llvm.sh) under the `scripts` directory:

```sh
source scripts/setup-vitis-hls-llvm.sh
```

This script is borrowed from the official Vitis HLS frontend [repository](https://github.com/Xilinx/HLS/blob/2020.2/plugins/setup-vitis-hls-llvm.sh). It is used to set environment variables to point essential build tools to the versions encapsulated in Vitis.

After running that script, please do check if the GCC toolchain correctly points to the version wrapped in Vitis, which should be 6.2.0 suppose you're using Vitis 2020.2.

```sh
gcc --version
# Should print "gcc (GCC) 6.2.0" 
```

#### CMake

You should also install your own [CMake](https://cmake.org/download/). Version 3.13.4 is the minimum required to work with the latest LLVM.

### Install LLVM

Just run the following script. It should take care of everything you need to install the [llvm](llvm) shipped with Phism.

```sh
./scripts/build-llvm.sh
```
