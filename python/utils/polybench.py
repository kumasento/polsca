""" Utitity functions.  """

import os
import glob
import shutil
import datetime
import pandas as pd
import subprocess
import functools
import itertools
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from collections import namedtuple
from timeit import default_timer as timer

POLYBENCH_DATASETS = ('MINI', 'SMALL', 'LARGE', 'EXTRALARGE')
POLYBENCH_EXAMPLES = ('2mm', '3mm', 'adi', 'atax', 'bicg', 'cholesky', 'correlation', 'covariance', 'deriche', 'doitgen', 'durbin', 'fdtd-2d', 'gemm', 'gemver',
                      'gesummv', 'gramschmidt', 'head-3d', 'jacobi-1D', 'jacobi-2D', 'lu', 'ludcmp', 'mvt', 'nussinov', 'seidel', 'symm', 'syr2k', 'syrk', 'trisolv', 'trmm')
RESOURCE_FIELDS = ('DSP', 'FF', 'LUT', 'BRAM_18K', 'URAM')
RECORD_FIELDS = ('name', 'latency', 'res_usage', 'res_avail')

Record = namedtuple('Record', RECORD_FIELDS)
Resource = namedtuple('Resource', RESOURCE_FIELDS)

# ----------------------- Utility functions ------------------------------------


def get_timestamp():
    """ Get the current timestamp. """
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def get_project_root():
    """ Get the root directory of the project. """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def matched(s, patterns):
    """ Check if string s matches any of the patterns. """
    if not patterns:
        return False
    for p in patterns:
        if p in s:
            return True
    return False


def get_single_file_with_ext(d, ext, includes=None):
    """ Find the single file under the current directory with a specific extension. """
    for f in os.listdir(d):
        if not f.endswith(ext):
            continue
        if includes and not matched(f, includes):
            continue
        return f

    return None

# ----------------------- Data record fetching functions -----------------------


def fetch_resource_usage(d, avail=False):
    """ Find the report file *.xml and return the resource usage estimation. """
    syn_report_dir = os.path.join(d, 'proj', 'solution1', 'syn', 'report')
    if not os.path.isdir(syn_report_dir):
        return None

    syn_report = get_single_file_with_ext(syn_report_dir, 'xml', ['kernel'])
    if not syn_report:
        return None

    syn_report = os.path.join(syn_report_dir, syn_report)
    if not os.path.isfile(syn_report):
        return None

    # Parse the XML report and find every resource usage (tags given by RESOURCE_FIELDS)
    root = ET.parse(syn_report).getroot()
    res_tag = 'Resources' if not avail else 'AvailableResources'
    return Resource(*[int(root.findtext('AreaEstimates/{}/{}'.format(res_tag, res)))
                      for res in RESOURCE_FIELDS])


def fetch_latency(d):
    """ Fetch the simulated latency, measured in cycles. """
    tb_sim_report_dir = os.path.join(d, 'tb', 'solution1', 'sim', 'report')
    if not os.path.isdir(tb_sim_report_dir):
        return None

    tb_sim_report = get_single_file_with_ext(tb_sim_report_dir, 'rpt')
    if not tb_sim_report:
        return None

    tb_sim_report = os.path.join(tb_sim_report_dir, tb_sim_report)
    if not os.path.isfile(tb_sim_report):
        return None

    latency = None
    with open(tb_sim_report, 'r') as f:
        for line in f.readlines():
            comps = [x.strip() for x in line.strip().split('|')]

            # there are 9 columns, +2 before & after |
            # the 2nd column should give PASS.
            if len(comps) == 11 and comps[2].upper() == 'PASS':
                latency = comps[-2]  # from the last column.

    # The report is malformed.
    if not latency:
        return None

    # Will raise error if latency is not an integer.
    return int(latency)


def process_directory(d):
    """ Process the result data within the given directory. Return a dictionary of all available data entries. """
    example_name = os.path.basename(d)
    return Record(example_name, fetch_latency(d), fetch_resource_usage(d), fetch_resource_usage(d, avail=True))


def process_pb_flow_result_dir(d):
    """ Process the result directory from pb-flow runs. """
    records = []

    # Each example should have their original .c/.h files. We will look for that.
    pattern = '{}/**/*.h'.format(d)
    for src_header_file in glob.glob(pattern, recursive=True):
        basename = os.path.basename(src_header_file)[:-2]  # skip '.h'
        if basename in POLYBENCH_EXAMPLES:
            records.append(process_directory(
                os.path.abspath(os.path.dirname(src_header_file))))

    return records


# ----------------------- Data processing ---------------------------

def expand_resource_field(field):
    """ Will turn things like "res_avail" to a list ['DSP_avail', 'FF_avail', ...] """
    if 'res_' not in field:
        return [field]
    avail = field.split('_')[-1]
    return ['{}_{}'.format(res, avail) for res in RESOURCE_FIELDS]


def flatten_record(record):
    """ Flatten a Record object into a list. """
    return list(itertools.chain(*[list(x) if isinstance(x, Resource) else [x] for x in record]))


def to_pandas(records):
    """ From processed records to pandas DataFrame. """
    cols = list(itertools.chain(*[expand_resource_field(field)
                                  for field in RECORD_FIELDS]))
    data = list([flatten_record(r) for r in records])
    data.sort(key=lambda x: x[0])

    # NOTE: dtype=object here prevents pandas converting integer to float.
    return pd.DataFrame(data=data, columns=cols, dtype=object)


# ----------------------- Benchmark runners ---------------------------

def discover_examples(d):
    """ Find examples in the given directory. """
    for root, _, files in os.walk(d):
        # There should be two files, one end with .h, the other with .c
        if len(files) != 2:
            continue
        if len(files[0]) <= 2 or len(files[1]) <= 2 or files[0][:-2] != files[1][:-2]:
            continue
        yield root


def get_phism_env():
    """ Get the Phism run-time environment. """
    root_dir = get_project_root()

    phism_env = os.environ.copy()
    phism_env['PATH'] = '{}:{}:{}'.format(
        os.path.join(root_dir, 'llvm', 'build', 'bin'),
        os.path.join(root_dir, 'build', 'bin'),
        phism_env['PATH'])
    phism_env['LD_LIBRARY_PATH'] = '{}:{}:{}:{}'.format(
        os.path.join(root_dir, 'llvm', 'build', 'lib'),
        os.path.join(root_dir, 'llvm', 'build', 'tools', 'mlir',
                     'tools', 'polymer', 'pluto', 'lib'),
        os.path.join(root_dir, 'build', 'lib'),
        phism_env['LD_LIBRARY_PATH'])

    return phism_env


def get_top_func(src_file):
    """ Get top function name. """
    return 'kernel_{}'.format(os.path.basename(os.path.dirname(src_file))).replace('-', '_')


PHISM_VITIS_TCL = '''
open_project -reset proj
add_files {dummy_src}
set_top {top_func}

open_solution -reset solution1
set_part "zynq"
create_clock -period "100MHz"
{config}

set ::LLVM_CUSTOM_CMD {{\$LLVM_CUSTOM_OPT -no-warn {src_file} -o \$LLVM_CUSTOM_OUTPUT}}

csynth_design

exit
'''

TBGEN_VITIS_TCL = '''
open_project -reset tb
add_files {{{src_dir}/{src_base}.c}} -cflags "-I {src_dir} -I {work_dir}/utilities -D {pb_dataset}_DATASET" -csimflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET"
add_files -tb {{{src_dir}/{src_base}.c {work_dir}/utilities/polybench.c}} -cflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET" -csimflags "-I {src_dir} -I {work_dir}/utilities -D{pb_dataset}_DATASET"
set_top {top_func}

open_solution -reset solution1
set_part "zynq"
create_clock -period "100MHz"
{config}

csim_design
csynth_design
cosim_design

exit
'''

COSIM_VITIS_TCL = '''
open_project tb

open_solution solution1

cosim_design

exit
'''


class PbFlow:
    """ Holds all the pb-flow functions.
        TODO: inherits this from PhismFlow.
    """

    def __init__(self, work_dir, options):
        """ Constructor. `work_dir` is the top of the polybench directory. """
        self.env = get_phism_env()
        self.root_dir = get_project_root()
        self.work_dir = work_dir
        self.cur_file = None
        self.options = options

    def run(self, src_file):
        """ Run the whole pb-flow on the src_file (*.c). """
        self.cur_file = src_file

        # The whole flow
        (self
         .compile_c()
         .preprocess()
         .extract_top_func()
         .polymer_opt()
         .lower_llvm()
         .vitis_opt()
         .run_vitis())

    def compile_c(self):
        """ Compile C code to MLIR using mlir-clang. """
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            '.c', '.mlir')

        subprocess.run(
            'sed -i "s/static//g" "{}"'.format(src_file), shell=True)
        subprocess.run([
            'mlir-clang',
            src_file,
            '-memref-fullrank',
            '-D', '{}_DATASET'.format(self.options.dataset),
            '-I={}'.format(os.path.join(
                self.root_dir, 'llvm', 'build', 'lib', 'clang', '13.0.0', 'include')),
            '-I={}'.format(os.path.join(self.work_dir, 'utilities'))],
            stdout=open(self.cur_file, 'w'),
            env=self.env)
        return self

    def preprocess(self):
        """ Do some preprocessing before extracting the top function. """
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            '.mlir', '.pre.mlir')
        subprocess.run(['mlir-opt', src_file, '-sccp', '-canonicalize'],
                       stdout=open(self.cur_file, 'w'), env=self.env)
        return self

    def extract_top_func(self):
        """Extract the top function and all the stuff it calls. """
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            '.mlir', '.kern.mlir')
        subprocess.run('phism-opt {} -extract-top-func="name={}"'.format(
            src_file, get_top_func(src_file)),
            shell=True,
            stdout=open(self.cur_file, 'w'),
            env=self.env)
        return self

    def polymer_opt(self):
        """ Run polymer optimization. """
        if not self.options.polymer:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            '.mlir', '.plmr.mlir')
        log_file = self.cur_file.replace('.mlir', '.log')

        subprocess.run(['polymer-opt',
                        src_file,
                        '-reg2mem',
                        '-insert-redundant-load',
                        '-extract-scop-stmt',
                        '-pluto-opt'],
                       stderr=open(log_file, 'w'),
                       stdout=open(self.cur_file, 'w'),
                       env=self.env)

        return self

    def lower_llvm(self):
        """ Lower from MLIR to LLVM. """
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            '.mlir', '.llvm')

        args = ['mlir-opt',
                src_file,
                '-lower-affine',
                '-inline',
                '-convert-scf-to-std',
                '-canonicalize',
                '-convert-std-to-llvm="use-bare-ptr-memref-call-conv=1"',
                '| mlir-translate -mlir-to-llvmir']

        subprocess.run(' '.join(args),
                       shell=True,
                       stdout=open(self.cur_file, 'w'),
                       env=self.env)

        return self

    def vitis_opt(self):
        """ Optimize LLVM IR for Vitis. """
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            '.llvm', '.vitis.llvm')

        args = [os.path.join(self.root_dir, 'llvm', 'build', 'bin', 'opt'),
                src_file,
                '-S',
                '-enable-new-pm=0',
                '-load "{}"'.format(os.path.join(self.root_dir,
                                    'build', 'lib', 'VhlsLLVMRewriter.so')),
                '-strip-debug',
                '-mem2arr',
                '-instcombine',
                '-xlnmath',
                '-xlnname',
                '-xlnanno',
                '-xlntop="{}"'.format(get_top_func(src_file)),
                '-strip-attr']

        subprocess.run(' '.join(args),
                       shell=True,
                       stdout=open(self.cur_file, 'w'),
                       env=self.env)

        return self

    def run_vitis(self):
        """ Run synthesize/testbench generation/co-simulation. """
        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)
        top_func = get_top_func(src_file)

        phism_vitis_tcl = os.path.join(base_dir, 'phism.tcl')
        tbgen_vitis_tcl = os.path.join(base_dir, 'tbgen.tcl')
        cosim_vitis_tcl = os.path.join(base_dir, 'cosim.tcl')

        run_config = 'config_bind -effort high'
        if self.options.debug:
            run_config = ''

        # Run Phism Vitis
        dummy_src = src_file.replace('.llvm', '.dummy.c')
        with open(dummy_src, 'w') as f:
            f.write('void {}() {{}}'.format(top_func))
        with open(phism_vitis_tcl, 'w') as f:
            f.write(PHISM_VITIS_TCL.format(
                src_file=src_file,
                dummy_src=dummy_src,
                top_func=top_func,
                config=run_config))

        phism_proc = subprocess.Popen(
            ['vitis_hls', phism_vitis_tcl],
            cwd=base_dir,
            stdout=open(os.path.join(
                base_dir, 'phism.vitis_hls.stdout.log'), 'w'),
            stderr=open(os.path.join(base_dir, 'phism.vitis_hls.stderr.log'), 'w'))

        # Run tbgen Vitis
        if self.options.cosim:
            with open(tbgen_vitis_tcl, 'w') as f:
                f.write(TBGEN_VITIS_TCL.format(
                    src_dir=base_dir,
                    src_base=os.path.basename(src_file).split('.')[0],
                    top_func=top_func,
                    work_dir=self.work_dir,
                    config=run_config,
                    pb_dataset=self.options.dataset))
            with open(cosim_vitis_tcl, 'w') as f:
                f.write(COSIM_VITIS_TCL)

            tbgen_proc = subprocess.Popen(
                ['vitis_hls', tbgen_vitis_tcl],
                cwd=base_dir,
                stdout=open(os.path.join(
                    base_dir, 'tbgen.vitis_hls.stdout.log'), 'w'),
                stderr=open(os.path.join(base_dir, 'tbgen.vitis_hls.stderr.log'), 'w'))

            # Allows phism_proc and tbgen_proc run in parallel.
            phism_proc.wait()
            tbgen_proc.wait()

            # TODO: add some sanity checks
            for f in glob.glob(os.path.join(base_dir, 'proj', 'solution1', 'syn', 'verilog', '*.v*')):
                shutil.copyfile(f, os.path.join(
                    base_dir, 'tb', 'solution1', 'sim', 'verilog'))

            # Run cosim for Phism in the end
            subprocess.run(
                ['vitis_hls', cosim_vitis_tcl],
                cwd=base_dir,
                stdout=open(os.path.join(
                    base_dir, 'cosim.vitis_hls.stdout.log'), 'w'),
                stderr=open(os.path.join(base_dir, 'cosim.vitis_hls.stderr.log'), 'w'))
        else:
            phism_proc.wait()

        return self


def pb_flow_process(d, work_dir, options):
    """ Process a single example. """
    flow = PbFlow(work_dir, options)
    src_file = os.path.join(d, get_single_file_with_ext(d, 'c'))

    start = timer()
    flow.run(src_file)
    end = timer()

    print('>>> Finished {:15s} elapsed: {:.6f} secs'.format(
        os.path.basename(d), end - start))


def pb_flow_runner(options):
    """ Run pb-flow with the provided arguments. """
    assert os.path.isdir(options.pb_dir)

    # Copy all the files from the source pb_dir to a target temporary directory.
    tmp_dir = os.path.join(get_project_root(), 'tmp', 'phism',
                           'pb-flow.{}'.format(get_timestamp()))
    shutil.copytree(options.pb_dir, tmp_dir)

    print('>>> Starting {} jobs ...'.format(options.job))

    start = timer()
    with Pool(options.job) as p:
        p.map(functools.partial(pb_flow_process, work_dir=tmp_dir, options=options),
              discover_examples(tmp_dir))
    end = timer()
    print('Elapsed time: {:.6f} sec'.format(end - start))
