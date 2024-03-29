#!/usr/bin/env python3

import argparse
import os
import shutil

from pyphism.phism_runner.options import PhismRunnerOptions
from pyphism.phism_runner.runner import PhismRunner
from pyphism.utils.helper import get_logger

logger = get_logger("rosetta")


def check_polygeist(work_dir: str, vitis_hls_root: str):
    """"""
    worklist = []
    for root, _, files in os.walk(work_dir):
        if root.endswith("sdsoc"):
            logger.info(f"Checking directory: {root}")
            source_file = next(f for f in files if f.endswith(".cpp"))
            if not source_file:
                logger.warn("No source file found.")

            logger.info(f"Found source_file: {source_file}")

            worklist.append((root, source_file))

    worklist = set(worklist)
    for root, source_file in worklist:
        top_func = source_file.split(".")[0]
        options = PhismRunnerOptions(
            key="{}-{}".format(root.split("/")[-1], source_file.split(".")[0]),
            source_file=os.path.join(root, source_file),
            source_dir=root,
            work_dir=root,
            sanity_check=False,
            top_func=top_func,
        )
        runner = PhismRunner(options=options)
        runner.set_cur_file()
        runner.polygeist_compile_c(
            flags=[
                "-I",
                os.path.abspath(root),
                "-I",
                f"{vitis_hls_root}/include",
                f"--function={top_func}",
            ],
            suffix=".cpp",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=str, help="Where the rosetta benchmark is.")
    parser.add_argument("--work-dir", type=str, help="Working directory.")
    parser.add_argument("--vitis-hls-root", type=str, help="Where is Vitis HLS.")
    args = parser.parse_args()

    shutil.copytree(args.source_dir, args.work_dir, dirs_exist_ok=True)

    check_polygeist(args.work_dir, vitis_hls_root=args.vitis_hls_root)


if __name__ == "__main__":
    main()
