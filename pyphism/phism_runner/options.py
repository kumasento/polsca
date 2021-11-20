from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PhismRunnerOptions:
    """Phism runner options.

    TODO: separate the CLI part (for all files) and the class part (for one file).
    """

    # --- Local
    key: str = ""  # The key to the current example.
    source_file: str = ""  # input source file
    top_func: str = ""  # top function name
    incl_funcs: str = ""  # include functions for SCoP extraction
    disabled: Optional[List[str]] = None  # disabled passes

    # --- Global
    cfg: str = ""  # configuration file.
    polymer: bool = False  # run with polymer
    loop_transforms: bool = False  # run phism loop transform
    array_partition: bool = False  # run phism array partition
    fold_if: bool = False  # run phism fold if
    skip_vitis: bool = False  # whether to skip the whole vitis flow
    cosim: bool = False  # whether to run cosim
    dry_run: bool = False  # only print out the commands to run
    sanity_check: bool = True  # run in sanity check mode
    source_dir: str = ""  # if specified, won't work with a single file
    work_dir: str = ""  # temporary workdir
    includes: Optional[List[str]] = None  # examples to include
    excludes: Optional[List[str]] = None  # examples to exclude
    jobs: int = 1  # Number of concurrent jobs
    tile_sizes: Optional[List[str]] = None  # tile size for each loop depth
