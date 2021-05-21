""" Utitity functions.  """

import os
import glob
from typing import NamedTuple
import pandas as pd
import itertools
import xml.etree.ElementTree as ET
from collections import namedtuple

POLYBENCH_EXAMPLES = ('2mm', '3mm', 'adi', 'atax', 'bicg', 'cholesky', 'correlation', 'covariance', 'deriche', 'doitgen', 'durbin', 'fdtd-2d', 'gemm', 'gemver',
                      'gesummv', 'gramschmidt', 'head-3d', 'jacobi-1D', 'jacobi-2D', 'lu', 'ludcmp', 'mvt', 'nussinov', 'seidel', 'symm', 'syr2k', 'syrk', 'trisolv', 'trmm')
RESOURCE_FIELDS = ('DSP', 'FF', 'LUT', 'BRAM_18K', 'URAM')
RECORD_FIELDS = ('name', 'latency', 'res_usage', 'res_avail')

Record = namedtuple('Record', RECORD_FIELDS)
Resource = namedtuple('Resource', RESOURCE_FIELDS)

# ----------------------- Utility functions ------------------------------------


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
