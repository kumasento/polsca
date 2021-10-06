#!/usr/bin/env python3

"""
Search for different Polyhedral codegen configurations.
"""

import argparse
import copy
import os
import shutil

import pandas as pd

from pyphism.polybench import pb_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pb_dir", type=str, help="Polybench directory")
    parser.add_argument("--work-dir", type=str, help="The temporary work directory.")
    parser.add_argument(
        "-e", "--examples", nargs="+", default=[], help="Polybench examples to run."
    )
    parser.add_argument("--cloogl-start", type=int, help="Start value for -cloogl")
    parser.add_argument("--cloogf-start", type=int, help="Start value for -cloogf")
    parser.add_argument("--cloogl-end", type=int, help="End value for -cloogl")
    parser.add_argument("--cloogf-end", type=int, help="End value for -cloogf")
    parser.add_argument(
        "-j", "--job", type=int, default=1, help="Number of parallel jobs (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        choices=pb_flow.POLYBENCH_DATASETS,
        default="MINI",
        help="Polybench dataset size. ",
    )
    args = parser.parse_args()

    options = pb_flow.PbFlowOptions(**pb_flow.filter_init_args(vars(args)))
    options.polymer = True
    options.cosim = False
    options.loop_transforms = True
    options.array_partition = True
    options.cleanup = False

    results = None
    for cloogl in [-1] + list(range(args.cloogl_start, args.cloogl_end + 1)):
        for cloogf in [-1] + list(range(args.cloogf_start, args.cloogf_end + 1)):
            work_dir = os.path.join(args.work_dir, f"cloogl-{cloogl}-f-{cloogf}")
            # if os.path.isdir(work_dir):
            #     shutil.rmtree(work_dir)

            options_ = copy.deepcopy(options)
            options_.cloogf = cloogf
            options_.cloogl = cloogl
            options_.work_dir = work_dir

            pb_flow.pb_flow_runner(options_, dump_report=False)

            for d in pb_flow.discover_examples(options_.work_dir, options_.examples):
                info = pb_flow.fetch_pipeline_info(d)
                name = os.path.basename(d).lower()

                df = pd.DataFrame(info)
                df.insert(0, "name", name)
                df.insert(1, "cloogl", cloogl)
                df.insert(2, "cloogf", cloogf)

                if results is None:
                    results = df
                else:
                    results = pd.concat((results, df))

    print(results)


if __name__ == "__main__":
    main()
