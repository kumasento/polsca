#!/usr/bin/env python3
# A python version of the old pb-flow. Should be way faster with parallelism!
#
# USE: cd $PHSIM \
#      && PYTHONPATH=$PWD \
#      && python3 scripts/pb-flow.py -c example/polybench

import argparse

from pyphism.polybench import pb_flow


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
        choices=pb_flow.POLYBENCH_DATASETS,
        default="MINI",
        help="Polybench dataset size. ",
    )
    parser.add_argument("--cleanup", action="store_true", help="Cleanup after run.")
    parser.add_argument(
        "--max-span", type=int, default=-1, help="Max spanning of the point loops."
    )
    parser.add_argument(
        "--split", type=str, default="NO_SPLIT", help="Statement split method."
    )
    parser.add_argument(
        "--improve-pipelining",
        action="store_true",
        help="Enable pipelining improvement",
    )
    parser.add_argument(
        "--loop-transforms",
        action="store_true",
        help="Enable loop transforms",
    )
    parser.add_argument(
        "--tile-sizes", nargs="+", default=[], help="Tile sizes for each loop nest."
    )
    parser.add_argument(
        "--array-partition", action="store_true", help="Use array partition."
    )
    parser.add_argument("--skip-vitis", action="store_true", help="Don't run Vitis.")
    parser.add_argument(
        "--skip-csim", action="store_true", help="Don't run tbgen (csim)."
    )
    parser.add_argument("--sanity-check", action="store_true", help="Run sanity check.")
    parser.add_argument("--cloogl", type=int, default=-1, help="-cloogl option")
    parser.add_argument("--cloogf", type=int, default=-1, help="-cloogf option")
    parser.add_argument(
        "--diamond-tiling", action="store_true", help="Use diamond tiling"
    )
    args = parser.parse_args()

    options = pb_flow.PbFlowOptions(**vars(args))

    print(f"Options: {options}")

    pb_flow.pb_flow_runner(options)


if __name__ == "__main__":
    main()
