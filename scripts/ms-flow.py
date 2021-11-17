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
    args = parser.parse_args()

    options = MsFlowOptions(**vars(args))
    ms_flow_runner(options)


if __name__ == "__main__":
    main()
