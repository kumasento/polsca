#!/usr/bin/env python3
# A python version of the old pb-flow. Should be way faster with parallelism!
#
# USE: cd $PHSIM \
#      && PYTHONPATH=$PWD \
#      && python3 scripts/pb-flow.py -c example/polybench

import argparse
import os
import subprocess
import sys

import python.utils.polybench as pb_utils


def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description="Run Polybench experiments")
    parser.add_argument("pb_dir", type=str, help="Polybench directory")
    parser.add_argument("--work-dir", type=str, help="The temporary work directory.")
    parser.add_argument(
        "-e", "--examples", nargs="+", default=[], help="Polybench examples to run."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only produce the commands to run."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "-p",
        "--polymer",
        action="store_true",
        help="Use Polymer to perform polyhedral transformation",
    )
    parser.add_argument(
        "-c", "--cosim", action="store_true", help="Enable co-simulation"
    )
    parser.add_argument(
        "-j", "--job", type=int, default=1, help="Number of parallel jobs (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        choices=pb_utils.POLYBENCH_DATASETS,
        default="MINI",
        help="Polybench dataset size. ",
    )
    parser.add_argument("--cleanup", action="store_true", help="Cleanup after run.")
    args = parser.parse_args()

    options = pb_utils.PbFlowOptions(**vars(args))

    print(f"Options: {options}")

    pb_utils.pb_flow_runner(options)


if __name__ == "__main__":
    main()
