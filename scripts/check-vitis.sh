#!/usr/bin/env bash

# Check whether Xilinx Vitis environment is set properly.

if [ -z "${XILINX_HLS}" ]; then
  echo "XILINX_HLS is empty. Have you forgotten 'source VITIS_HLS/settings64.sh'?"
  exit 1
fi

GCC_PATH="$(which gcc)"
if [ "${GCC_PATH}" != "${XILINX_HLS}/tps/lnx64/gcc-6.2.0/bin/gcc" ]; then
  echo "You should point gcc to the version inside Vitis."
  echo "Have you forgotten to 'source scripts/setup-vitis-hls-llvm.sh'?"
  exit 1
fi
