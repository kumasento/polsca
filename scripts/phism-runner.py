#!/usr/bin/env python3


import argparse

from pyphism.phism_runner.options import PhismRunnerOptions
from pyphism.phism_runner.runner import PhismRunner


def main():
    parser = argparse.ArgumentParser(description="Phism main runner")
    parser.add_argument("source_file", type=str)
    parser.add_argument(
        "--work-dir", type=str, help="Temporary directory to store intermediate files."
    )
    parser.add_argument("--top-func", type=str, help="Top function name.")
    parser.add_argument(
        "-p", "--polymer", action="store_true", help="Run with polymer."
    )
    parser.add_argument(
        "--loop-transforms",
        "--lt",
        action="store_true",
        help="Run with phism loop transforms",
    )
    args = parser.parse_args()

    runner = PhismRunner(options=PhismRunnerOptions(**vars(args)))
    runner.run()


if __name__ == "__main__":
    main()
