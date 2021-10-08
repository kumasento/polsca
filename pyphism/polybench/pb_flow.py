""" Utitity functions for polybench evaluation.  """

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
from dataclasses import dataclass
from multiprocessing import Pool
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import pyphism.utils.helper as helper
from pyphism.polybench.utils import vhdl

POLYBENCH_DATASETS = ("MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE")
POLYBENCH_EXAMPLES = (
    "2mm",
    "3mm",
    "adi",
    "atax",
    "bicg",
    "cholesky",
    "correlation",
    "covariance",
    "deriche",
    "doitgen",
    "durbin",
    "fdtd-2d",
    "floyd-warshall",
    "gemm",
    "gemver",
    "gesummv",
    "gramschmidt",
    "heat-3d",
    "jacobi-1d",
    "jacobi-2d",
    "lu",
    "ludcmp",
    "mvt",
    "nussinov",
    "seidel-2d",
    "symm",
    "syr2k",
    "syrk",
    "trisolv",
    "trmm",
)
RESOURCE_FIELDS = ("DSP", "FF", "LUT", "BRAM_18K", "URAM")
RECORD_FIELDS = (
    "name",
    "run_status",
    "latency",
    "res_usage",
    "res_avail",
)
RUN_STATUS_FIELDS = ("status",)

PHISM_VITIS_STEPS = ("phism", "tbgen", "cosim")

Record = namedtuple("Record", RECORD_FIELDS)
Resource = namedtuple("Resource", RESOURCE_FIELDS)
RunStatus = namedtuple("RunStatus", RUN_STATUS_FIELDS)


@dataclass
class PbFlowOptions:
    """An interface for the CLI options."""

    pb_dir: str
    job: int = 1
    polymer: bool = False
    # CLooG options
    cloogf: int = -1
    cloogl: int = -1

    dataset: str = "MINI"
    cleanup: bool = False
    debug: bool = False
    work_dir: str = ""
    dry_run: bool = False
    examples: List[str] = POLYBENCH_EXAMPLES
    split: str = "NO_SPLIT"  # other options: "SPLIT", "HEURISTIC"
    loop_transforms: bool = False
    constant_args: bool = True
    improve_pipelining: bool = False
    max_span: int = -1
    tile_sizes: Optional[List[int]] = None
    array_partition: bool = False
    cosim: bool = False
    skip_vitis: bool = False
    skip_csim: bool = False  # Given cosim = True, you can still turn down csim.
    sanity_check: bool = False  # Run pb-flow in sanity check mode

    def __post_init__(self):
        if self.sanity_check:
            # Disable the Vitis steps.
            self.cosim = False
            self.skip_vitis = True
            self.skip_csim = True


def filter_init_args(args: Dict[str, Any]) -> Dict[str, Any]:
    opt = PbFlowOptions(pb_dir="")
    return {k: v for k, v in args.items() if hasattr(opt, k)}


# ----------------------- Utility functions ------------------------------------


def get_timestamp():
    """Get the current timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_project_root():
    """Get the root directory of the project."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def matched(s, patterns):
    """Check if string s matches any of the patterns."""
    if not patterns:
        return False
    for p in patterns:
        if p in s:
            return True
    return False


def get_single_file_with_ext(d, ext, includes=None, excludes=None):
    """Find the single file under the current directory with a specific extension."""
    for f in os.listdir(d):
        if not f.endswith(ext):
            continue
        if includes and not matched(f, includes):
            continue
        if excludes and matched(f, excludes):
            continue
        return f

    return None


def get_vitis_log(d, step, stream):
    """Return the file path to a specific Vitis log."""
    assert step in PHISM_VITIS_STEPS
    assert stream in ("stdout", "stderr")

    return os.path.join(
        d, "{step}.vitis_hls.{stream}.log".format(step=step, stream=stream)
    )


# ----------------------- Data record fetching functions -----------------------


def fetch_resource_usage(d, avail=False):
    """Find the report file *.xml and return the resource usage estimation."""
    syn_report_dir = os.path.join(d, "tb", "solution1", "syn", "report")
    if not os.path.isdir(syn_report_dir):
        return None

    syn_report = get_single_file_with_ext(syn_report_dir, "xml", ["kernel"], ["PE"])
    if not syn_report:
        return None

    syn_report = os.path.join(syn_report_dir, syn_report)
    if not os.path.isfile(syn_report):
        return None

    # Parse the XML report and find every resource usage (tags given by RESOURCE_FIELDS)
    root = ET.parse(syn_report).getroot()
    res_tag = "Resources" if not avail else "AvailableResources"
    return Resource(
        *[
            int(root.findtext("AreaEstimates/{}/{}".format(res_tag, res)))
            for res in RESOURCE_FIELDS
        ]
    )


def fetch_latency(d: str, csim: bool = False):
    """Fetch the simulated latency, measured in cycles."""
    tb_sim_report_dir = os.path.join(
        d, "tb" if not csim else "tb.csim", "solution1", "sim", "report"
    )
    if not os.path.isdir(tb_sim_report_dir):
        return None

    tb_sim_report = get_single_file_with_ext(tb_sim_report_dir, "rpt")
    if not tb_sim_report:
        return None

    tb_sim_report = os.path.join(tb_sim_report_dir, tb_sim_report)
    if not os.path.isfile(tb_sim_report):
        return None

    latency = None
    with open(tb_sim_report, "r") as f:
        for line in reversed(f.readlines()):
            if latency:
                break
            comps = [x.strip() for x in line.strip().split("|")]

            # there are 9 columns, +2 before & after |
            # the 2nd column should give PASS.
            if len(comps) == 11 and comps[2].upper() == "PASS":
                latency = comps[-2]  # from the last column.

    # The report is malformed.
    if not latency:
        return None

    try:
        # Will raise error if latency is not an integer.
        return int(latency)
    except:
        return None


def fetch_syn_latency(d):
    """Fetch latency measured in the synthesis phase."""
    syn_report_dir = os.path.join(d, "proj", "solution1", "syn", "report")
    if not os.path.isdir(syn_report_dir):
        return None

    syn_report = get_single_file_with_ext(syn_report_dir, "xml", ["kernel"], ["PE"])
    if not syn_report:
        return None

    syn_report = os.path.join(syn_report_dir, syn_report)
    if not os.path.isfile(syn_report):
        return None

    # Parse the XML report and find every resource usage (tags given by RESOURCE_FIELDS)
    root = ET.parse(syn_report).getroot()
    latency = root.findtext(
        "PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency"
    )
    try:
        return int(latency)
    except:
        return None


def fetch_run_status(d):
    """Gather the resulting status of each stage."""

    def parse_synth_log(fp):
        if not os.path.isfile(fp):
            return "NO_LOG"
        else:
            with open(fp, "r") as f:
                has_error = any(
                    ("Synthesizability check failed." in line) for line in f.readlines()
                )
                if has_error:
                    return "CANNOT_SYNTH"
        return "SUCCESS"

    def parse_cosim_log(fp):
        if not os.path.isfile(fp):
            return "NO_LOG"
        else:
            with open(fp, "r") as f:
                has_error = any(
                    ("co-simulation finished: FAIL" in line) for line in f.readlines()
                )
                if has_error:
                    return "COSIM_FAILED"
        return "SUCCESS"

    return RunStatus(
        # parse_synth_log(get_vitis_log(d, "phism", "stdout")),
        parse_cosim_log(get_vitis_log(d, "tbgen", "stdout")),
        # parse_cosim_log(get_vitis_log(d, "cosim", "stdout")),
    )


def fetch_pipeline_info(d: str, proj_name: str = "tb") -> Dict[str, List[Any]]:
    """Find the pipeline II result from the provided project directory."""
    syn_report_dir = os.path.join(d, proj_name, "solution1", "syn", "report")
    if not os.path.isdir(syn_report_dir):
        return None

    syn_report = os.path.join(syn_report_dir, "csynth.xml")
    if not os.path.isfile(syn_report):
        return None

    # Parse the XML report and find every resource usage (tags given by RESOURCE_FIELDS)
    data = defaultdict(list)
    root = ET.parse(syn_report).getroot()

    def process(el: Optional[ET.Element], module_name: str):
        if el is None:
            return

        pipeline_ii = el.findtext("PipelineII")
        if pipeline_ii is not None:
            name = el.findtext("Name")

            data["module_name"].append(module_name)
            data["loop_name"].append(name)
            data["pipeline_ii"].append(pipeline_ii)

            return

        for child in el.getchildren():
            process(child, module_name)

    for el in root.findall("ModuleInformation/Module"):
        module_name = el.findtext("Name")
        loops = el.find("PerformanceEstimates/SummaryOfLoopLatency")
        process(loops, module_name)

    return data


def process_directory(d):
    """Process the result data within the given directory. Return a dictionary of all available data entries."""
    example_name = os.path.basename(d)
    return Record(
        example_name,
        fetch_run_status(d),
        fetch_latency(d),
        # fetch_latency(d, csim=True),
        # fetch_syn_latency(d),
        fetch_resource_usage(d),
        fetch_resource_usage(d, avail=True),
    )


def process_pb_flow_result_dir(d: str, options: PbFlowOptions):
    """Process the result directory from pb-flow runs."""
    records = []
    assert os.path.isdir(d)

    # Each example should have their original .c/.h files. We will look for that.
    pattern = "{}/**/*.h".format(d)
    for src_header_file in glob.glob(pattern, recursive=True):
        basename = os.path.basename(src_header_file)[:-2]  # skip '.h'
        if basename in options.examples:
            records.append(
                process_directory(os.path.abspath(os.path.dirname(src_header_file)))
            )

    return records


def filter_success(df):
    """Filter success rows."""
    return df[df["status"] == "SUCCESS"]


# ----------------------- Data processing ---------------------------


def expand_resource_field(field):
    """Will turn things like "res_avail" to a list ['DSP_avail', 'FF_avail', ...]"""
    if "res_" not in field:
        return [field]
    avail = field.split("_")[-1]
    return ["{}_{}".format(res, avail) for res in RESOURCE_FIELDS]


def expand_field(field):
    """Turn a nested namedtuple into a flattened one."""
    if "res_" in field:
        return expand_resource_field(field)
    if "run_status" in field:
        return RUN_STATUS_FIELDS
    return [field]


def is_list_record(x):
    return isinstance(x, (Resource, RunStatus))


def flatten_record(record):
    """Flatten a Record object into a list."""
    return list(
        itertools.chain(*[list(x) if is_list_record(x) else [x] for x in record])
    )


def to_pandas(records):
    """From processed records to pandas DataFrame."""
    cols = list(itertools.chain(*[expand_field(field) for field in RECORD_FIELDS]))
    data = list([flatten_record(r) for r in records])
    data.sort(key=lambda x: x[0])

    # NOTE: dtype=object here prevents pandas converting integer to float.
    return pd.DataFrame(data=data, columns=cols, dtype=object)


# ----------------------- Cosim utilities ---------------------------


@dataclass
class ApMemoryInterface:
    name: str
    ports: List[str]

    def get_name_without_partition(self) -> str:
        return self.name.split("_")[0]

    def get_num_ports(self) -> int:
        return len(set([port[-1] for port in self.ports]))

    def is_read_only(self, port_id: int) -> bool:
        return f"we{port_id}" not in self.ports

    def is_write_only(self, port_id: int) -> bool:
        return f"q{port_id}" not in self.ports

    def is_read_write(self, port_id: int) -> bool:
        return f"q{port_id}" in self.ports and f"d{port_id}" in self.ports


def get_module_parameters(file: str, module_name: str) -> List[str]:
    """Read the module definition into parameter lists."""
    with open(file, "r") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    start_line = next(i for i, l in enumerate(lines) if f"module {module_name}" in l)
    end_line = next(i for i, l in enumerate(lines) if ");" in l)

    params = (" ".join(line for line in lines[start_line + 1 : end_line])).split(",")
    return [param.strip() for param in params if param.strip()]


def get_autotb_parameters(file: str) -> List[str]:
    """Read interface from autotb files."""
    assert os.path.isfile(file)
    assert file.endswith(".autotb.v")

    with open(file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    start_line = next(
        i for i, l in enumerate(lines) if f"`AUTOTB_DUT `AUTOTB_DUT_INST(" in l
    )
    assert start_line >= 0 and start_line < len(lines)

    end_line = next(i for i, l in enumerate(lines) if ");" in l and i > start_line)
    assert end_line >= 0 and end_line < len(lines)

    # Deal with things like -
    # .ap_clk(ap_clk),
    # .ap_rst(ap_rst),

    conns = (" ".join(line for line in lines[start_line + 1 : end_line + 1])).split(",")
    conns = [conn.strip() for conn in conns]

    params = []
    for conn in conns:
        if conn.endswith(");"):
            conn = conn[:-2]
        assert conn[0] == "." and "(" in conn and conn[-1] == ")"
        param = conn.split("(")[0][1:]
        assert param == conn.split("(")[1][:-1]
        params.append(param)

    return params


def get_memory_interfaces(params: List[str]):
    """Parse memory interfaces from the module params."""
    interfaces = OrderedDict()
    for param in params:
        prefix = "_".join(param.split("_")[:-1])
        if prefix not in interfaces:
            interfaces[prefix] = []
        if param.startswith("ap") or "_" not in param:
            continue
        interfaces[prefix].append(param.split("_")[-1])

    return [
        ApMemoryInterface(name, ports)
        for name, ports in interfaces.items()
        if "address0" in ports
    ]


@dataclass
class CosimFixStrategy:
    phism_directives: List[str]
    tbgen_directives: List[str]
    phism_mem_interfaces: List[ApMemoryInterface]
    tbgen_mem_interfaces: List[ApMemoryInterface]

    def empty(self):
        return not self.phism_directives and not self.tbgen_directives


def is_read_write_conflict(
    src_mem: ApMemoryInterface, dst_mem: ApMemoryInterface, port_id: int
) -> bool:
    return (src_mem.is_read_only(port_id) and dst_mem.is_write_only(port_id)) or (
        dst_mem.is_read_only(port_id) and src_mem.is_write_only(port_id)
    )


def is_cosim_interface_matched(
    src_mems: List[ApMemoryInterface], dst_mems: List[ApMemoryInterface]
) -> bool:
    if len(src_mems) != len(dst_mems):
        return False

    for src, dst in zip(src_mems, dst_mems):
        if src.get_num_ports() != dst.get_num_ports():
            return False
        if set(src.ports) != set(dst.ports):
            return False

    return True


def get_cosim_fix_strategy(
    kernel_name: str,
    src_mems: List[ApMemoryInterface],
    dst_mems: List[ApMemoryInterface],
    before_partition: bool = True,
) -> CosimFixStrategy:
    if len(src_mems) != len(dst_mems):
        raise RuntimeError("The number of ap_memory interfaces should be the same.")
    if [mem.name for mem in src_mems] != [mem.name for mem in dst_mems]:
        raise RuntimeError("The name of the interfaces should be the same.")

    # Determine whether we can fix this.
    strategy = CosimFixStrategy(
        phism_directives=[],
        tbgen_directives=[],
        phism_mem_interfaces=src_mems,
        tbgen_mem_interfaces=dst_mems,
    )

    # Iterate every memory interface to see if there is any chance for fixing them.
    for src_mem, dst_mem in zip(src_mems, dst_mems):
        dst_mem_name = (
            dst_mem.name
            if not before_partition
            else dst_mem.get_name_without_partition()
        )
        # If any memory interface from the source uses single port, while the target uses dual ports,
        # we will modify the TCL for Phism.
        if src_mem.get_num_ports() == 1 and dst_mem.get_num_ports() == 2:
            strategy.tbgen_directives.append(
                f"set_directive_interface -mode ap_memory -storage_type ram_1p {kernel_name} {dst_mem_name}"
            )
        elif src_mem.get_num_ports() == 2 and dst_mem.get_num_ports() == 1:
            strategy.tbgen_directives.append(
                f"set_directive_interface -mode ap_memory -storage_type ram_2p {kernel_name} {dst_mem_name}"
            )
        elif src_mem.get_num_ports() == dst_mem.get_num_ports():
            num_ports = src_mem.get_num_ports()
            if num_ports == 2:
                # Make sure the dst_mem is 1 write n read.
                # TODO: is this condition enough for detection?
                if src_mem.is_read_only(1) and dst_mem.is_read_write(1):
                    strategy.tbgen_directives.append(
                        f"set_directive_interface -mode ap_memory -storage_type ram_1wnr {kernel_name} {dst_mem_name}"
                    )
                elif (
                    src_mem.is_read_write(0)
                    and src_mem.is_read_write(1)  # Phism is T2P
                    and (
                        dst_mem.is_read_only(0) or dst_mem.is_read_only(1)
                    )  # tbgen is not
                ):
                    strategy.tbgen_directives.append(
                        f"set_directive_interface -mode ap_memory -storage_type ram_t2p {kernel_name} {dst_mem_name}"
                    )
                elif is_read_write_conflict(
                    src_mem, dst_mem, 0
                ) and is_read_write_conflict(src_mem, dst_mem, 1):
                    strategy.tbgen_directives.append(
                        f"set_directive_interface -mode ap_memory -storage_type ram_1wnr {kernel_name} {dst_mem_name}"
                    )

    strategy.tbgen_directives = list(set(strategy.tbgen_directives))
    strategy.phism_directives = list(set(strategy.phism_directives))

    return strategy


def fix_cosim_kernels(dir: str) -> CosimFixStrategy:
    """Fix issues with co-simulation.
    Returns directives for (source, destination).
    """

    dir = os.path.abspath(dir)  # canonicalize path
    kernel_name = f"kernel_{os.path.basename(dir)}"

    src_proj_dir = os.path.join(dir, "proj", "solution1")
    assert os.path.isdir(src_proj_dir)

    dst_proj_dir = os.path.join(dir, "tb.backup", "solution1")
    assert os.path.isdir(dst_proj_dir)

    src_kernel = os.path.join(src_proj_dir, "syn", "verilog", f"{kernel_name}.v")
    assert os.path.isfile(src_kernel)

    dst_kernel = os.path.join(dst_proj_dir, "syn", "verilog", f"{kernel_name}.v")
    assert os.path.isfile(dst_kernel)

    src_params = get_module_parameters(src_kernel, kernel_name)
    dst_params = get_module_parameters(dst_kernel, kernel_name)

    return get_cosim_fix_strategy(
        kernel_name,
        get_memory_interfaces(src_params),
        get_memory_interfaces(dst_params),
    )


def insert_directives(directives: List[str], file: str, insertion_point: str):
    """Insert directives within the target file before the insertion point."""
    with open(file, "r") as f:
        lines = f.readlines()
    assert lines

    lines = [l.strip() for l in lines]

    pos = next(i for i, l in enumerate(lines) if insertion_point in l)
    assert pos >= 0 and pos < len(lines)

    lines = lines[:pos] + directives + lines[pos:]
    with open(file, "w") as f:
        f.write("\n".join(lines))


def is_cosim_setup(file: str):
    with open(file, "r") as f:
        lines = f.readlines()

    return any(("cosim_design" in line and "setup" in line) for line in lines)


def toggle_cosim_setup(file: str):
    """Toggle the -setup option for cosim_design."""
    with open(file, "r") as f:
        lines = f.readlines()
    assert lines

    lines = [l.strip() for l in lines]
    pos = next(i for i, l in enumerate(lines) if "cosim_design" in l)
    assert pos >= 0 and pos < len(lines)

    if "-setup" in lines[pos]:
        lines[pos] = lines[pos].replace("-setup", "")
    else:
        lines[pos] += " -setup"

    with open(file, "w") as f:
        f.write("\n".join(lines))


def comment_out_cosim(file: str):
    with open(file, "r") as f:
        lines = f.readlines()
    assert lines

    lines = [l.strip() for l in lines]
    pos = next(i for i, l in enumerate(lines) if "cosim_design" in l)
    assert pos >= 0 and pos < len(lines)

    lines[pos] = f"# {lines[pos]}"

    with open(file, "w") as f:
        f.write("\n".join(lines))


# ----------------------- Benchmark runners ---------------------------


def discover_examples(d: str, examples: Optional[List[str]] = None) -> List[str]:
    """Find examples in the given directory."""
    if not examples:
        examples = POLYBENCH_EXAMPLES

    return sorted(
        [
            root
            for root, _, _ in os.walk(d)
            if os.path.basename(root).lower() in examples
        ]
    )


def get_phism_env():
    """Get the Phism run-time environment."""
    root_dir = get_project_root()

    phism_env = os.environ.copy()
    phism_env["PATH"] = ":".join(
        [
            os.path.join(root_dir, "polygeist", "llvm-project", "build", "bin"),
            os.path.join(root_dir, "polygeist", "build", "mlir-clang"),
            os.path.join(root_dir, "polymer", "build", "bin"),
            os.path.join(root_dir, "build", "bin"),
            phism_env["PATH"],
        ]
    )
    phism_env["LD_LIBRARY_PATH"] = "{}:{}:{}:{}".format(
        os.path.join(root_dir, "polygeist", "llvm-project", "build", "lib"),
        os.path.join(root_dir, "polymer", "build", "pluto", "lib"),
        os.path.join(root_dir, "build", "lib"),
        phism_env["LD_LIBRARY_PATH"],
    )

    return phism_env


def get_top_func(src_file):
    """Get top function name."""
    return "kernel_{}".format(os.path.basename(os.path.dirname(src_file))).replace(
        "-", "_"
    )


def get_top_func_param_names(src_file, pb_dir, llvm_dir=None):
    """From the given C file, we try to extract the top function's parameter list.
    This will be useful for Vitis LLVM rewrite."""

    def is_func_decl(item, name):
        return item["kind"] == "FunctionDecl" and item["name"] == name

    top_func = get_top_func(src_file)
    clang_path = "clang"
    if llvm_dir:
        clang_path = os.path.join(llvm_dir, "build", "bin", "clang")

    # Get the corresponding AST in JSON.
    proc = subprocess.Popen(
        [
            clang_path,
            src_file,
            "-Xclang",
            "-ast-dump=json",
            "-fsyntax-only",
            "-I{}".format(os.path.join(pb_dir, "utilities")),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    data = json.loads(proc.stdout.read())

    # First find the top function declaration entry.
    top_func_decls = list(
        filter(functools.partial(is_func_decl, name=top_func), data["inner"])
    )
    assert len(top_func_decls) == 1, "Should be a single declaration for top."
    top_func_decl = top_func_decls[0]

    # Then get all ParmVarDecl.
    parm_var_decls = filter(
        lambda x: x["kind"] == "ParmVarDecl", top_func_decl["inner"]
    )
    return [decl["name"] for decl in parm_var_decls]


PHISM_VITIS_TCL = """
open_project -reset proj
add_files {dummy_src}
set_top {top_func}

open_solution -reset solution1
set_part "xqzu29dr-ffrf1760-1-i"
create_clock -period "100MHz"
config_compile -pipeline_loops 1
{config}

set ::LLVM_CUSTOM_CMD {{$LLVM_CUSTOM_OPT -no-warn {src_file} -o $LLVM_CUSTOM_OUTPUT}}

csynth_design
# export_design -flow syn -rtl vhdl -format ip_catalog
exit
"""

TBGEN_VITIS_TCL = """
open_project -reset tb
add_files {{{src_dir}/{src_base}.c}} -cflags "-I {src_dir} -I {work_dir}/utilities -D {pb_dataset}_DATASET" -csimflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET"
add_files -tb {{{src_dir}/{src_base}.c {work_dir}/utilities/polybench.c}} -cflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET" -csimflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET"
set_top {top_func}

open_solution -reset solution1
set_part "xqzu29dr-ffrf1760-1-i"
create_clock -period "100MHz"
{config}

csim_design
csynth_design
cosim_design -rtl vhdl 

exit
"""

COSIM_VITIS_TCL = """
open_project tb

open_solution solution1

cosim_design -rtl vhdl 

exit
"""


class PbFlow:
    """Holds all the pb-flow functions.
    TODO: inherits this from PhismFlow.
    """

    def __init__(self, work_dir: str, options: PbFlowOptions):
        """Constructor. `work_dir` is the top of the polybench directory."""
        self.env = get_phism_env()
        self.root_dir = get_project_root()
        self.work_dir = work_dir
        self.cur_file = None
        self.c_source = None
        self.options = options

        self.status = 0
        self.errmsg = "No Error"

        # Logger
        self.logger = logging.getLogger("pb-flow")
        self.logger.setLevel(logging.DEBUG)

    def run(self, src_file):
        """Run the whole pb-flow on the src_file (*.c)."""
        self.cur_file = src_file
        self.c_source = src_file  # Will be useful in some later stages

        base_dir = os.path.dirname(src_file)

        # Setup logging
        log_file = os.path.join(base_dir, f"pb-flow.log")
        if os.path.isfile(log_file):
            os.remove(log_file)

        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        # The whole flow
        try:
            (
                self.generate_tile_sizes()
                .dump_test_data()
                .compile_c()
                .preprocess()
                .sanity_check()
                .split_statements()
                .extract_top_func()
                .polymer_opt()
                .sanity_check()
                .constant_args()
                .sanity_check()
                .loop_transforms()
                .sanity_check()
                .array_partition()
                .sanity_check()
                .scop_stmt_inline()
                .sanity_check()
                .lower_llvm()
                .vitis_opt()
                .write_tb_tcl_by_llvm()
                # .run_vitis_on_phism()
                .run_vitis()
                # .backup_csim_results()
                # .copy_design_from_phism_to_tb()
                # .run_cosim()
            )
        except Exception as e:
            self.status = 1
            self.errmsg = e

            # Log stack
            self.logger.error(traceback.format_exc())

    def run_command(
        self, cmd: str = "", cmd_list: Optional[List[str]] = None, **kwargs
    ):
        """Single entry for running a command."""
        if "cwd" not in kwargs:
            kwargs.update({"cwd": os.path.dirname(self.cur_file)})

        if cmd_list:
            cmd_list = [cmd for cmd in cmd_list if cmd]
            cmd_ = " \\\n\t".join(cmd_list)
            self.logger.debug(f"{cmd_}")
            if self.options.dry_run:
                print(" ".join(cmd_list))
                return
            proc = subprocess.run(cmd_list, **kwargs)
        else:
            self.logger.debug(f"{cmd}")
            if self.options.dry_run:
                print(cmd)
                return
            proc = subprocess.run(cmd, **kwargs)

        cmd_str = cmd if cmd else " ".join(cmd_list)
        if proc.returncode != 0:
            raise RuntimeError(f"{cmd_str} failed.")

        return proc

    def get_program_abspath(self, program: str) -> str:
        """Get the absolute path of a program."""
        return str(
            subprocess.check_output(["which", program], env=self.env), "utf-8"
        ).strip()

    def get_golden_out_file(self) -> str:
        path = os.path.basename(self.cur_file)
        return os.path.join(
            os.path.dirname(self.cur_file), path.split(".")[0] + ".golden.out"
        )

    def dump_test_data(self):
        """Compile and dump test data for sanity check."""
        if not self.options.sanity_check:
            return self

        out_file = self.get_golden_out_file()
        exe_file = self.cur_file.replace(".c", ".exe")
        self.run_command(
            cmd=" ".join(
                [
                    self.get_program_abspath("clang"),
                    "-D",
                    f"{self.options.dataset}_DATASET",
                    "-D",
                    "POLYBENCH_DUMP_ARRAYS",
                    "-I",
                    os.path.join(self.work_dir, "utilities"),
                    "-I",
                    os.path.join(
                        self.root_dir,
                        "polygeist",
                        "llvm-project",
                        "build",
                        "lib",
                        "clang",
                        "14.0.0",
                        "include",
                    ),
                    "-lm",
                    self.cur_file,
                    os.path.join(self.work_dir, "utilities", "polybench.c"),
                    "-o",
                    exe_file,
                ]
            ),
            shell=True,
            env=self.env,
        )
        self.run_command(
            cmd=exe_file,
            stderr=open(out_file, "w"),
            env=self.env,
        )

        return self

    def generate_tile_sizes(self):
        """Generate the tile.sizes file that Pluto will read."""
        base_dir = os.path.dirname(self.cur_file)
        tile_file = os.path.join(base_dir, "tile.sizes")

        if not self.options.tile_sizes:
            if os.path.isfile(tile_file):
                os.remove(tile_file)
            return self

        with open(tile_file, "w") as f:
            for tile in self.options.tile_sizes:
                f.write(f"{tile}\n")

        return self

    def compile_c(self):
        """Compile C code to MLIR using mlir-clang."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(".c", ".mlir")

        self.run_command(cmd=f'sed -i "s/static//g" {src_file}', shell=True)
        self.run_command(
            cmd=" ".join(
                [
                    self.get_program_abspath("mlir-clang"),
                    src_file,
                    "-memref-fullrank",
                    "-S",
                    "-O0",
                    "-D",
                    f"{self.options.dataset}_DATASET",
                    "-D",
                    "POLYBENCH_DUMP_ARRAYS",
                    "-I",
                    os.path.join(
                        self.root_dir,
                        "polygeist",
                        "llvm-project",
                        "build",
                        "lib",
                        "clang",
                        "14.0.0",
                        "include",
                    ),
                    "-I",
                    os.path.join(self.work_dir, "utilities"),
                ]
            ),
            stdout=open(self.cur_file, "w"),
            shell=True,
            env=self.env,
        )
        return self

    def sanity_check(self):
        """Sanity check the current file."""
        if not self.options.sanity_check:
            return self

        assert self.cur_file.endswith(".mlir"), "Should be an MLIR file."

        out_file = self.cur_file.replace(".mlir", ".out")
        self.run_command(
            cmd=" ".join(
                [
                    self.get_program_abspath("mlir-opt"),
                    "-lower-affine",
                    "-convert-scf-to-std",
                    "-convert-memref-to-llvm",
                    "-convert-std-to-llvm",
                    self.cur_file,
                    "|",
                    self.get_program_abspath("mlir-translate"),
                    "-mlir-to-llvmir",
                    "|",
                    self.get_program_abspath("opt"),
                    "-O3",
                    "|",
                    self.get_program_abspath("lli"),
                ]
            ),
            shell=True,
            env=self.env,
            stderr=open(out_file, "w"),
        )

        self.run_command(
            cmd_list=["diff", self.get_golden_out_file(), out_file],
            stdout=open(out_file.replace(".out", ".diff"), "w"),
        )

        return self

    def preprocess(self):
        """Do some preprocessing before extracting the top function."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".pre.mlir"
        )
        self.run_command(
            cmd_list=[
                self.get_program_abspath("mlir-opt"),
                "-sccp" if not self.options.sanity_check else "",
                "-canonicalize",
                src_file,
            ],
            stderr=open(
                os.path.join(
                    os.path.dirname(self.cur_file),
                    self.cur_file.replace(".mlir", ".log"),
                ),
                "w",
            ),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )
        return self

    def split_statements(self):
        """Use Polymer to split statements."""
        if self.options.split == "NO_SPLIT":
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", f".{self.options.split.lower()}.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        self.run_command(
            cmd_list=[
                self.get_program_abspath("polymer-opt"),
                src_file,
                "-reg2mem",
                (
                    "-annotate-splittable"
                    if self.options.split == "SPLIT"
                    else "-annotate-heuristic"
                ),
                "-scop-stmt-split",
                "-canonicalize",
            ],
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def extract_top_func(self):
        """Extract the top function and all the stuff it calls."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".kern.mlir"
        )

        keepall = "keepall" if self.options.sanity_check else ""

        log_file = self.cur_file.replace(".mlir", ".log")
        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            f'-extract-top-func="name={get_top_func(src_file)} {keepall}"',
        ]
        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )
        return self

    def polymer_opt(self):
        """Run polymer optimization."""
        if not self.options.polymer:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".plmr.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        passes = []
        if self.options.split == "NO_SPLIT":  # The split stmt has applied -reg2mem
            passes += [
                "-reg2mem",
            ]
        passes += [
            "-extract-scop-stmt",
            f'-pluto-opt="cloogf={self.options.cloogf} cloogl={self.options.cloogl}"',
            "-debug",
        ]

        self.run_command(
            cmd=" ".join(
                [
                    self.get_program_abspath("polymer-opt"),
                    src_file,
                ]
                + passes
            ),
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            shell=True,
            env=self.env,
        )

        return self

    def scop_stmt_inline(self):
        """Inline scop.stmt."""
        if self.options.loop_transforms:
            self.logger.debug(
                "Skipped scop.stmt inline since there're completed already in loop transforms."
            )
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".si.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            "-scop-stmt-inline",
            "-debug-only=loop-transforms",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def loop_transforms(self):
        """Run Phism loop transforms."""
        if not self.options.loop_transforms:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".lt.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            f'-loop-transforms="max-span={self.options.max_span}"',
            "-loop-redis-and-merge",
            "-fold-if",
            "-debug-only=loop-transforms",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def array_partition(self):
        """Run Phism array partition transforms."""
        if not self.options.array_partition:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".ap.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        array_partition_file = os.path.join(
            os.path.dirname(self.cur_file), "array_partition.txt"
        )
        if os.path.isfile(array_partition_file):
            os.remove(array_partition_file)

        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            '-simple-array-partition="dumpFile flatten"',
            "-debug-only=array-partition",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def constant_args(self):
        """Run Phism constant args."""
        if not self.options.constant_args:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".ca.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            f'-replace-constant-arguments="name={get_top_func(src_file)}"',
            "-canonicalize",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def lower_llvm(self):
        """Lower from MLIR to LLVM."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(".mlir", ".llvm")

        memref_option = (
            f"use-bare-ptr-memref-call-conv={0 if self.options.sanity_check else 1}"
        )
        convert_std_to_llvm = f'-convert-std-to-llvm="{memref_option}"'

        args = [
            self.get_program_abspath("mlir-opt"),
            src_file,
            "-lower-affine",
            "-convert-scf-to-std",
            "-canonicalize",
            convert_std_to_llvm,
            f"| {self.get_program_abspath('mlir-translate')} -mlir-to-llvmir",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def vitis_opt(self):
        """Optimize LLVM IR for Vitis."""
        if self.options.sanity_check:
            self.logger.debug("Skipped --vitis-opt since in --sanity-check.")
            return self
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".llvm", ".vitis.llvm"
        )
        log_file = self.cur_file.replace(".llvm", ".log")

        xln_names = get_top_func_param_names(
            self.c_source, self.work_dir, llvm_dir=os.path.join(self.root_dir, "llvm")
        )

        # Whether array partition has been successful.
        xln_ap_enabled = os.path.isfile(
            os.path.join(os.path.dirname(self.cur_file), "array_partition.txt")
        )

        args = [
            os.path.join(self.root_dir, "llvm", "build", "bin", "opt"),
            src_file,
            "-S",
            "-enable-new-pm=0",
            '-load "{}"'.format(
                os.path.join(self.root_dir, "build", "lib", "VhlsLLVMRewriter.so")
            ),
            "-strip-debug",
            "-mem2arr",
            "-instcombine",
            "-xlnmath",
            "-xlnname",
            "-xlnanno",
            '-xlntop="{}"'.format(get_top_func(src_file)),
            '-xlnnames="{}"'.format(",".join(xln_names)),
            "-xlnunroll" if self.options.loop_transforms else "",
            "-xlnram2p",
            "-xlnarraypartition" if self.options.array_partition else "",
            "-xln-ap-flattened",
            "-xln-ap-enabled" if xln_ap_enabled else "",
            "-strip-attr",
            "-debug",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stdout=open(self.cur_file, "w"),
            stderr=open(log_file, "w"),
            env=self.env,
        )

        return self

    def write_tb_tcl_by_llvm(self):
        """Generate the tbgen TCL file from LLVM passes."""
        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)
        top_func = get_top_func(src_file)

        # Whether array partition has been successful.
        xln_ap_enabled = os.path.isfile(os.path.join(base_dir, "array_partition.txt"))

        tbgen_vitis_tcl = os.path.join(base_dir, "tbgen.tcl")

        tb_tcl_log = "write_tb_tcl_by_llvm.log"

        # Write the TCL for TBGEN.
        args = [
            os.path.join(self.root_dir, "llvm", "build", "bin", "opt"),
            src_file,
            "-S",
            "-enable-new-pm=0",
            '-load "{}"'.format(
                os.path.join(self.root_dir, "build", "lib", "VhlsLLVMRewriter.so")
            ),
            f'-xlntop="{top_func}"',
            "-xlntbgen",
            "-xln-ap-flattened",
            "-xln-ap-enabled" if xln_ap_enabled else "",
            f"-xlntbdummynames={base_dir}/dummy.cpp",
            f'-xlntbtclnames="{tbgen_vitis_tcl}"',
            f'-xlnllvm="{src_file}"',
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stdout=open(tb_tcl_log, "w"),
            env=self.env,
        )

        return self

    def run_vitis_on_phism(self):
        """Just run vitis_hls on the LLVM generated from Phism."""
        # DEPRECATED
        if self.options.skip_vitis:
            self.logger.warn("Vitis won't run since --skip-vitis has been set.")
            return self
        if self.options.cosim:
            self.logger.warn("Vitis won't run since --cosim has been set.")
            return self

        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)
        top_func = get_top_func(src_file)

        phism_vitis_tcl = os.path.join(base_dir, "phism.tcl")
        run_config = "config_bind -effort high"
        if self.options.debug:
            run_config = ""

        # Generate dummy C code as the interface for the top function.
        dummy_src = src_file.replace(".llvm", ".dummy.c")
        with open(dummy_src, "w") as f:
            f.write("void {}() {{}}".format(top_func))

        # Write the TCL for Phism.
        with open(phism_vitis_tcl, "w") as f:
            phism_run_config = [str(run_config)]
            f.write(
                PHISM_VITIS_TCL.format(
                    src_file=src_file,
                    dummy_src=dummy_src,
                    top_func=top_func,
                    config="\n".join(phism_run_config),
                )
            )

        log_file = os.path.join(base_dir, "phism.vitis_hls.stdout.log")

        # Clean up old results
        shutil.rmtree(os.path.join(base_dir, "proj"), ignore_errors=True)
        if os.path.isfile(log_file):
            os.remove(log_file)

        if self.options.dry_run:
            return self

        self.run_command(
            cmd_list=["vitis_hls", phism_vitis_tcl],
            stdout=open(log_file, "w"),
            stderr=open(os.path.join(base_dir, "phism.vitis_hls.stderr.log"), "w"),
            env=self.env,
        )

        return self

    def run_vitis(self, force_skip=False):
        """Run the tbgen.tcl file. Assuming the Tcl file has been written."""
        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)

        tbgen_vitis_tcl = os.path.join(base_dir, "tbgen.tcl")
        assert os.path.isfile(tbgen_vitis_tcl), f"{tbgen_vitis_tcl} should exist."

        if self.options.skip_csim or force_skip:
            self.logger.warn("CSim is set to be skipped.")
            if not is_cosim_setup(tbgen_vitis_tcl):
                self.logger.debug("Toggled -setup to cosim_design.")
                toggle_cosim_setup(tbgen_vitis_tcl)

        if not self.options.cosim:
            self.logger.warn("Cosim won't run due to the input setting.")
            comment_out_cosim(tbgen_vitis_tcl)

        if self.options.dry_run:
            return self

        tb_dir = os.path.join(base_dir, "tb")
        if os.path.isdir(tb_dir):
            shutil.rmtree(tb_dir)
            self.logger.debug(f"Removed old {tb_dir}")
        log_file = os.path.join(base_dir, "tbgen.vitis_hls.stdout.log")
        if os.path.isfile(log_file):
            os.remove(log_file)

        self.run_command(
            cmd_list=["vitis_hls", tbgen_vitis_tcl],
            stdout=open(log_file, "w"),
            stderr=open(os.path.join(base_dir, "tbgen.vitis_hls.stderr.log"), "w"),
            env=self.env,
        )

        return self

    def backup_csim_results(self):
        """Create a backup for the csim results."""
        if not self.options.cosim:
            return self
        # TODO: make this --dry-run compatible
        base_dir = os.path.dirname(self.cur_file)
        tbgen_dir = os.path.join(base_dir, "tb")
        assert os.path.isdir(
            tbgen_dir
        ), f"tbgen_dir={tbgen_dir} isn't there, please don't skip csim in this case."

        csim_dir = os.path.join(base_dir, "tb.csim")
        if os.path.isdir(csim_dir):
            self.logger.debug(f"csim_dir={csim_dir} exists, deleting it ...")
            shutil.rmtree(csim_dir)

        # Backup the tbgen (csim) results.
        shutil.copytree(tbgen_dir, csim_dir)

        return self

    def copy_design_from_phism_to_tb(self, try_fix=True):
        """Move design files from Phism output to the testbench directory."""
        if not self.options.cosim:
            return self

        # TODO: make this --dry-run compatible
        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)
        top_func = get_top_func(src_file)

        # ------------------------------- Paths
        # The design files generated by Phism
        phism_syn_vhdl_dir = os.path.join(
            base_dir, "proj", "solution1", "impl", "ip", "hdl", "vhdl"
        )
        # The ip files used in the design files by Phism
        phism_syn_ip_dir = os.path.join(
            base_dir, "proj", "solution1", "impl", "ip", "hdl", "ip"
        )
        # The test bench files
        tbgen_sim_vhdl_dir = os.path.join(base_dir, "tb", "solution1", "sim", "vhdl")

        # Sanity check
        assert os.path.isdir(phism_syn_vhdl_dir), f"{phism_syn_vhdl_dir} doens't exist."
        assert os.path.isdir(phism_syn_ip_dir), f"{phism_syn_ip_dir} doens't exist."
        assert os.path.isdir(tbgen_sim_vhdl_dir), f"{tbgen_sim_vhdl_dir} doens't exist."

        # ------------------------------- Copy and paste files
        # Copy and paste the design files.
        design_files = glob.glob(os.path.join(phism_syn_vhdl_dir, "*.*"))
        assert design_files, "There should exist design files."
        for f in design_files:
            shutil.copy(f, tbgen_sim_vhdl_dir)
        self.logger.debug(
            f"Design files found and copied: \n" + "\n".join(design_files)
        )
        # Copy and paste the ip design files.
        ip_design_files = glob.glob(os.path.join(phism_syn_ip_dir, "*.v"))
        assert ip_design_files, "There should exist design files."
        for f in ip_design_files:
            # Prepend the `timescale setting to the beginning of each ip design file.
            helper.prepend_to_file(
                shutil.copy(f, tbgen_sim_vhdl_dir), "`timescale 1ns/1ps"
            )

        self.logger.debug(f"IP files found and copied: \n" + "\n".join(ip_design_files))

        # ------------------------------- Update the top design
        phism_top = os.path.join(phism_syn_vhdl_dir, f"{top_func}.vhd")
        assert os.path.isfile(phism_top), f"The top module {phism_top} should exist."
        autotb = os.path.join(tbgen_sim_vhdl_dir, f"{top_func}.autotb.vhd")
        assert os.path.isfile(autotb), f"The autotb file {autotb} should exist."
        newtop = os.path.join(tbgen_sim_vhdl_dir, f"{top_func}.vhd")
        vhdl.update_source_by_testbench(
            phism_top,
            autotb,
            newtop,
            top_func,
            logger=self.logger,
        )

        # ------------------------------- Overwrite the prj file.
        # Overwrite the project file list to include files from Phism
        vhdl.create_prj_file(tbgen_sim_vhdl_dir, top_func)

        # TODO: Overwrite the test vectors in tv/

        # ------------------------------- Remove cached design files
        # Otherwise the cosimulation may continue with errors using the old design files
        xsim_dir = os.path.join(tbgen_sim_vhdl_dir, "xsim.dir")
        if os.path.isdir(xsim_dir):
            shutil.rmtree(xsim_dir)
            self.logger.debug(f"{xsim_dir} has been removed.")

        return self

    def run_cosim(self):
        """Run cosim.tcl"""
        if not self.options.cosim:
            self.logger.debug("cosim is skipped since --cosim has not been set.")
            return self

        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)
        sim_dir = os.path.join(base_dir, "tb", "solution1", "sim", "verilog")

        self.run_command(
            cmd="bash run_xsim.sh",
            shell=True,
            stdout=open(os.path.join(base_dir, "cosim.stdout.log"), "w"),
            stderr=open(os.path.join(base_dir, "cosim.stderr.log"), "w"),
            cwd=sim_dir,
        )

        return self


def pb_flow_process(d: str, work_dir: str, options: PbFlowOptions):
    """Process a single example."""
    # Make sure the example directory and the work directory are both absolute paths.
    # TODO: make it clear what is d.
    d = os.path.abspath(d)
    work_dir = os.path.abspath(work_dir)

    flow = PbFlow(work_dir, options)
    src_file = os.path.join(d, os.path.basename(d) + ".c")

    start = timer()
    flow.run(src_file)
    end = timer()

    if not options.dry_run:
        print(
            '>>> Finished {:15s} elapsed: {:.6f} secs   Status: {}  Error: "{}"'.format(
                os.path.basename(d), (end - start), flow.status, flow.errmsg
            )
        )


def pb_flow_dump_report(options: PbFlowOptions):
    """Dump report to the work_dir."""
    df = to_pandas(process_pb_flow_result_dir(options.work_dir, options))
    print("\n")
    print(df)
    print("\n")

    df.to_csv(os.path.join(options.work_dir, f"pb-flow.report.{get_timestamp()}.csv"))


def pb_flow_runner(options: PbFlowOptions, dump_report: bool = True):
    """Run pb-flow with the provided arguments."""
    assert os.path.isdir(options.pb_dir)

    if not options.examples:
        options.examples = POLYBENCH_EXAMPLES

    # Copy all the files from the source pb_dir to a target temporary directory.
    if not options.work_dir:
        options.work_dir = os.path.join(
            get_project_root(), "tmp", "phism", "pb-flow.{}".format(get_timestamp())
        )
    if not os.path.exists(options.work_dir):
        shutil.copytree(options.pb_dir, options.work_dir)

    print(
        ">>> Starting {} jobs (work_dir={}) ...".format(options.job, options.work_dir)
    )

    start = timer()
    with Pool(options.job) as p:
        # TODO: don't pass work_dir as an argument. Reuse it.
        p.map(
            functools.partial(
                pb_flow_process, work_dir=options.work_dir, options=options
            ),
            discover_examples(options.work_dir, examples=options.examples),
        )
    end = timer()
    print("Elapsed time: {:.6f} sec".format(end - start))

    # Will only dump report if Vitis has been run.
    if dump_report and not options.skip_vitis:
        print(">>> Dumping report ... ")
        pb_flow_dump_report(options)


# ------------------------------ Plotting -------------------------------
