#!/usr/bin/env python3

import argparse

from pyphism.machsuite.ms_flow import MsFlowOptions, ms_flow_runner


def main():
    parser = argparse.ArgumentParser(description="MachSuite runner.")
    parser.add_argument("source_dir", type=str, help="MachSuite directory.")
    parser.add_argument("--work-dir", type=str, help="Working directory.")
    parser.add_argument(
        "--jobs", "-j", default=1, type=int, help="Number of concucrrent jobs."
    )
    parser.add_argument(
        "--polymer", "-p", action="store_true", help="Whether to run polymer."
    )
    parser.add_argument(
        "--tile-sizes", nargs="+", default=[], help="Tile sizes for each loop nest."
    )
    parser.add_argument(
        "--includes",
        "--incls",
        nargs="+",
        default=[],
        help="MachSuite examples to run.",
    )
    parser.add_argument(
        "--excludes",
        "--excls",
        nargs="+",
        default=[],
        help="MachSuite examples not to run.",
    )
    parser.add_argument("--cfg", type=str, help="Configuration file.")
    parser.add_argument(
        "-c", "--cosim", action="store_true", help="Enable co-simulation"
    )
    parser.add_argument(
        "--loop-transforms",
        "--lt",
        action="store_true",
        help="Enable loop transforms",
    )
    parser.add_argument(
        "--array-partition", "--ap", action="store_true", help="Use array partition."
    )
    args = parser.parse_args()

    options = MsFlowOptions(**vars(args))
    ms_flow_runner(options)


if __name__ == "__main__":
    main()
