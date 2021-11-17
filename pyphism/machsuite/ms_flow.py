""" MachSuite runner """
import copy
import datetime
import functools
import glob
import itertools
import json
import logging
import os
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass, field
from multiprocessing import Pool
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

from pyphism.phism_runner.options import PhismRunnerOptions
from pyphism.phism_runner.runner import PhismRunner
from pyphism.utils.helper import get_logger, get_project_root, get_timestamp

# Default MachSuite examples to run.
MACHSUITE_EXAMPLES = [
    "aes/aes",
    "backprop/backprop",
    "bfs/bulk",
    "bfs/queue",
    "fft/strided",
    "fft/transpose",
    "gemm/blocked",
    "gemm/ncubed",
    "kmp/kmp",
    "md/grid",
    "md/knn",
    "nw/nw",
    "sort/merge",
    "sort/radix",
    "spmv/crs",
    "spmv/ellpack",
    "stencil/stencil2d",
    "stencil/stencil3d",
    "viterbi/viterbi",
]


@dataclass
class MsFlowOptions(PhismRunnerOptions):
    pass


class MsFlowRunner(PhismRunner):
    """Customised phism runner for MachSuite."""

    def get_name(self):
        path = os.path.abspath(self.options.work_dir)
        return "/".join(path.split("/")[-2:])

    def setup_logger(self):
        """Setup self.logger."""
        log_file = os.path.join(
            self.options.work_dir, f"ms-flow-runner.{get_timestamp()}.log"
        )
        self.logger = get_logger(
            f"ms-flow-runner {self.get_name()}", log_file, console=False
        )

    def run(self):
        self.logger.info("Started ms-flow-runner ...")

        # self.cur_file will be the entry for the following passes.
        self.cur_file = os.path.join(
            os.path.abspath(self.options.work_dir),
            os.path.basename(self.options.source_file),
        )
        self.logger.info(f"The input source file: {self.cur_file}")
        assert os.path.isfile(self.cur_file)

        self.c_source = self.cur_file

        try:
            (
                self.polygeist_compile_c(
                    flags=[
                        f"--function='{self.options.top_func}'",
                        "-I",
                        os.path.join(self.options.work_dir, "..", "..", "common"),
                    ]
                )
                # .mlir_preprocess()
                # .phism_extract_top_func()
                # .polymer_opt()
                # .phism_fold_if()
                # .phism_loop_transforms()
                # .phism_array_partition()
                # .lower_scf()
                # .lower_llvm()
                # .phism_vitis_opt()
                # .phism_dump_tcl()
                # .run_vitis()
            )
        except Exception as e:
            self.logger.error(traceback.format_exc())


def discover_examples(
    work_dir: str,
    incls: Optional[List[str]] = None,
    excls: Optional[List[str]] = None,
) -> List[str]:
    if not incls:
        incls = MACHSUITE_EXAMPLES
    if excls:
        incls = [x for x in incls if x not in excls]

    results = []
    for root, _, _ in os.walk(work_dir):
        path = os.path.abspath(root)
        name = "/".join(path.split("/")[-2:])
        if name in incls:
            results.append(path)

    return sorted(results)


def parse_ms_tcl(path: str, options: MsFlowOptions, logger: logging.Logger):
    """Parse MachSuite Tcl file to get necessary info."""
    tcl_path = os.path.join(path, "hls.tcl")
    assert os.path.isfile(tcl_path)

    with open(tcl_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("set_top"):
            top_func = line.split()[-1]
        if line.startswith("add_files") and ".c" in line and "-tb" not in line:
            source_file = line.split()[1]

    assert top_func
    assert source_file
    logger.info(f"Top function: {top_func:30s} source file: {source_file}")

    options = copy.deepcopy(options)
    options.top_func = top_func
    options.source_file = source_file
    options.work_dir = path
    options.sanity_check = False

    return options


def ms_flow_process(options: MsFlowOptions, logger: logging.Logger = None):
    """Process a single MachSuite example."""
    runner = MsFlowRunner(options)
    runner.run()


def ms_flow_runner(options: MsFlowOptions):

    if not options.work_dir:
        options.work_dir = os.path.join(
            get_project_root(), "tmp", "phism", "ms-flow.{}".format(get_timestamp())
        )
    if not os.path.exists(options.work_dir):
        shutil.copytree(options.ms_dir, options.work_dir)

    logger = get_logger(
        "ms-flow-runner",
        os.path.join(options.work_dir, f"ms-flow.{get_timestamp()}.log"),
    )

    logger.info(f"Options: {options}")
    examples_to_run = discover_examples(
        options.work_dir, incls=options.includes, excls=options.excludes
    )
    logger.info("Discovered examples: \n\t{}".format("\n\t".join(examples_to_run)))

    logger.info(f"Preparing runner options")
    example_options = [parse_ms_tcl(path, options, logger) for path in examples_to_run]

    logger.info(f"Starting {options.jobs} concurrent jobs")
    start = timer()
    with Pool(options.jobs) as p:
        p.map(
            functools.partial(ms_flow_process, logger=logger),
            example_options,
        )
    end = timer()
    logger.info("Elapsed time: {:.6f} secs".format(end - start))
