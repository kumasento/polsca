#!/usr/bin/env python3
# Fix the kernel mis-match between the Phism generated and the C generated code.

import argparse

from pyphism.polybench import pb_flow


def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description="Fix kernels for co-simulation.")
    parser.add_argument("dir", type=str, help="Where is the work directory.")

    args = parser.parse_args()

    strategy = pb_flow.fix_cosim_kernels(args.dir)
    print(strategy.phism_mem_interfaces)
    print(strategy.tbgen_mem_interfaces)
    print(strategy.phism_directives)
    print(strategy.tbgen_directives)


if __name__ == "__main__":
    main()
