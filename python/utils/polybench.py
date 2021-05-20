""" Utitity functions.  """

import os
import glob
from collections import namedtuple

POLYBENCH_EXAMPLES = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'cholesky', 'correlation', 'covariance', 'deriche', 'doitgen', 'durbin', 'fdtd-2d', 'gemm', 'gemver',
                      'gesummv', 'gramschmidt', 'head-3d', 'jacobi-1D', 'jacobi-2D', 'lu', 'ludcmp', 'mvt', 'nussinov', 'seidel', 'symm', 'syr2k', 'syrk', 'trisolv', 'trmm']


Record = namedtuple('Record', ['name', 'latency'])

# ----------------------- Utility functions ------------------------------------


def get_single_file_with_ext(d, ext):
    """ Find the single file under the current directory with a specific extension. """
    for f in os.listdir(d):
        if f.endswith(ext):
            return f
    return None

# ----------------------- Data record fetching functions -----------------------


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
    return Record(example_name, fetch_latency(d))


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
