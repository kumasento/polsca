from dataclasses import dataclass


@dataclass
class PhismRunnerOptions:
    source_file: str = ""  # input source file
    work_dir: str = ""  # temporary workdir
    dry_run: bool = False  # only print out the commands to run
    sanity_check: bool = True  # run in sanity check mode
    top_func: str = ""  # top function name
    polymer: bool = False  # run with polymer
    loop_transforms: bool = False  # run phism loop transform
    fold_if: bool = False  # run phism fold if
