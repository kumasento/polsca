""" PhismRunner definition. """

import inspect
import logging
import os
import shutil
import subprocess
import traceback
from typing import List, Optional

import colorlog

from pyphism.phism_runner.options import PhismRunnerOptions
from pyphism.utils import helper


def get_project_root():
    """Get the root directory of the project."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


class PhismRunner:
    """A wrapper for phism-related transforms."""

    def __init__(self, options: PhismRunnerOptions):
        self.options = options
        self.root_dir = get_project_root()
        self.env = get_phism_env()
        # Will be later properly instantiated.
        self.cur_file = None
        self.c_source = None
        self.logger = None
        self.passes = []

        self.setup_work_dir()
        self.setup_logger()

    def setup_work_dir(self):
        """Instantiate the work directory."""
        if not os.path.isdir(self.options.work_dir):
            os.makedirs(self.options.work_dir, exist_ok=True)

        # Copy the source file and its subdirectories.
        source_dir = os.path.dirname(self.options.source_file)
        shutil.copytree(source_dir, self.options.work_dir, dirs_exist_ok=True)

    def setup_logger(self):
        """Setup self.logger."""
        self.logger = logging.getLogger("phism-runner")
        self.logger.setLevel(logging.DEBUG)

        log_file = os.path.join(
            self.options.work_dir, f"phism-runner.{helper.get_timestamp()}.log"
        )
        if os.path.isfile(log_file):
            os.remove(log_file)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s[%(asctime)s][%(name)s][%(levelname)s]%(reset)s"
                + " %(message_log_color)s%(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
                secondary_log_colors={"message": {"ERROR": "red", "CRITICAL": "red"}},
            )
        )
        self.logger.addHandler(ch)

    def run(self):
        self.logger.info("Started phism-runner ...")

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
                self.dump_test_data()
                .polygeist_compile_c()
                .mlir_preprocess()
                .phism_extract_top_func()
                .polymer_opt()
                .phism_fold_if()
                .phism_loop_transforms()
                .phism_array_partition()
                .lower_scf()
                .lower_llvm()
                .phism_vitis_opt()
                .phism_dump_tcl()
                .run_vitis()
            )
        except Exception as e:
            self.logger.error(traceback.format_exc())

    def run_command(
        self, cmd: str = "", cmd_list: Optional[List[str]] = None, **kwargs
    ):
        """Single entry for running a command."""
        if "cwd" not in kwargs:
            kwargs.update({"cwd": os.path.dirname(self.cur_file)})

        self.logger.info(f" --> Calling from {inspect.stack()[1].function}()")

        if cmd_list:
            cmd_list = [cmd for cmd in cmd_list if cmd]
            cmd_ = " ".join(cmd_list)
            self.logger.debug(f"Run command:\n\t{cmd_}")
            if self.options.dry_run:
                print(" ".join(cmd_list))
                return
            proc = subprocess.run(cmd_list, **kwargs)
        else:
            self.logger.debug(f"Run command:\n\t{cmd}")
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
        """Get the file name for the golden output."""
        path = os.path.basename(self.cur_file)
        return os.path.join(
            os.path.dirname(self.cur_file), path.split(".")[0] + ".golden.out"
        )

    # ---------------------------- Passes -------------------------------------

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
                    "-o",
                    exe_file,
                ]
            ),
            shell=True,
            env=self.env,
        )

        # Run and capture the program output.
        self.run_command(
            cmd=exe_file,
            stderr=open(out_file, "w"),
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
                    "-convert-arith-to-llvm",
                    "-reconcile-unrealized-casts",
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
            cmd=" ".join(["diff", self.get_golden_out_file(), out_file]),
            shell=True,
            stdout=open(out_file.replace(".out", ".diff"), "w"),
        )

        self.logger.info("Sanity check OK!")

        return self

    def polygeist_compile_c(self):
        """Compile C code to MLIR using mlir-clang."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(".c", ".mlir")

        self.run_command(cmd=f'sed -i "s/static//g" {src_file}', shell=True)
        self.run_command(
            cmd=" ".join(
                [
                    self.get_program_abspath("mlir-clang"),
                    src_file,
                    "-raise-scf-to-affine",
                    "-memref-fullrank",
                    "-S",
                    "-O0",
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
                ]
            ),
            stdout=open(self.cur_file, "w"),
            shell=True,
            env=self.env,
        )
        return self.sanity_check()

    def mlir_preprocess(self):
        """Do some preprocessing before extracting the top function."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".p.mlir"
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
        return self.sanity_check()

    def phism_extract_top_func(self):
        """Extract the top function and all the stuff it calls."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".ext.mlir"
        )

        log_file = self.cur_file.replace(".mlir", ".log")
        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            f'-extract-top-func="name={self.options.top_func} keepall=1"',
            "-split-non-affine='max-loop-depth=5 top-only=1'",
        ]
        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )
        return self.sanity_check()

    def polymer_opt(self):
        """Run polymer optimization."""
        if not self.options.polymer:
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".poly.mlir"
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        self.run_command(
            cmd=" ".join(
                [
                    self.get_program_abspath("polymer-opt"),
                    src_file,
                    # f"-annotate-scop='functions={self.options.top_func}'",
                    "-fold-scf-if",
                    "-reg2mem",
                    "-extract-scop-stmt",
                    f"-pluto-opt",
                    "-debug",
                ]
            ),
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            shell=True,
            env=self.env,
        )

        return self.sanity_check()

    def phism_loop_transforms(self):
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
            f"-loop-transforms",
            "-loop-redis-and-merge",
            "-debug-only=loop-transforms",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self.sanity_check()

    def phism_fold_if(self):
        """Run Phism -fold-if."""
        if not self.options.fold_if:
            return self

        src_file, self.cur_file = (
            self.cur_file,
            self.cur_file.replace(".mlir", ".fi.mlir"),
        )
        log_file = self.cur_file.replace(".mlir", ".log")

        args = [
            self.get_program_abspath("phism-opt"),
            src_file,
            "-scop-stmt-inline",
            "-eliminate-affine-load-store",
            "-fold-if",
            "-debug-only=fold-if",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self.sanity_check()

    def phism_array_partition(self):
        """Run phism -array-partition."""
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
            "-array-partition",
            "-canonicalize",
            "-simplify-partition-access",
            "-canonicalize",
            "-debug-only=array-partition",
            "-verify-each=1",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stderr=open(log_file, "w"),
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def lower_scf(self):
        """Lower to SCF first."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".mlir", ".scf.mlir"
        )

        self.run_command(
            cmd_list=[
                self.get_program_abspath("phism-opt"),
                src_file,
                "-lower-affine",
            ],
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self.sanity_check() if not self.options.sanity_check else self

    def lower_llvm(self):
        """Lower from MLIR to LLVM."""
        src_file, self.cur_file = self.cur_file, self.cur_file.replace(".mlir", ".llvm")

        memref_option = f"use-bare-ptr-memref-call-conv=1"
        convert_std_to_llvm = f'-convert-std-to-llvm="{memref_option}"'

        args = [
            self.get_program_abspath("mlir-opt"),
            src_file,
            "-convert-scf-to-std",
            "-convert-memref-to-llvm",
            convert_std_to_llvm,
            "-convert-arith-to-llvm",
            "-reconcile-unrealized-casts",
            f"| {self.get_program_abspath('mlir-translate')} -mlir-to-llvmir",
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stdout=open(self.cur_file, "w"),
            env=self.env,
        )

        return self

    def phism_vitis_opt(self):
        """Optimize LLVM IR for Vitis."""
        if self.options.skip_vitis:
            self.logger.info("Skipped vitis opt.")
            return self

        src_file, self.cur_file = self.cur_file, self.cur_file.replace(
            ".llvm", ".vitis.llvm"
        )
        log_file = self.cur_file.replace(".llvm", ".log")

        xln_names = helper.get_param_names(
            self.options.top_func,
            self.c_source,
            clang_path=self.get_program_abspath("clang"),
        )
        args = [
            os.path.join(
                self.root_dir, "polygeist", "llvm-project", "build", "bin", "opt"
            ),
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
            '-xlntop="{}"'.format(self.options.top_func),
            '-xlnnames="{}"'.format(",".join(xln_names)),
            # "-xlnunroll" if self.options.loop_transforms else "",
            "-xlnram2p",
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

    def phism_dump_tcl(self):
        """Generate the tbgen TCL file from LLVM passes."""
        if self.options.skip_vitis:
            self.logger.info("Skipped dump tcl.")
            return self

        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)
        top_func = self.options.top_func

        tbgen_vitis_tcl = os.path.join(base_dir, "tbgen.tcl")

        log_file = self.cur_file.replace(".llvm", ".tbgen.log")

        # Write the TCL for TBGEN.
        args = [
            os.path.join(
                self.root_dir, "polygeist", "llvm-project", "build", "bin", "opt"
            ),
            src_file,
            "-S",
            "-enable-new-pm=0",
            '-load "{}"'.format(
                os.path.join(self.root_dir, "build", "lib", "VhlsLLVMRewriter.so")
            ),
            f'-xlntop="{top_func}"',
            "-xlntbgen",
            f"-xlntbdummynames={base_dir}/dummy.cpp",
            f'-xlntbtclnames="{tbgen_vitis_tcl}"',
            f'-xlnllvm="{src_file}"',
        ]

        self.run_command(
            cmd=" ".join(args),
            shell=True,
            stdout=open(log_file, "w"),
            env=self.env,
        )

        return self

    def run_vitis(self):
        """Run the tbgen.tcl file. Assuming the Tcl file has been written."""
        if self.options.skip_vitis:
            self.logger.info("Skipped run_vitis.")
            return self

        src_file = self.cur_file
        base_dir = os.path.dirname(src_file)

        tbgen_vitis_tcl = os.path.join(base_dir, "tbgen.tcl")
        assert os.path.isfile(tbgen_vitis_tcl), f"{tbgen_vitis_tcl} should exist."
        if not self.options.cosim:
            self.logger.warn("Cosim won't run due to the input setting.")
            helper.comment_out_cosim(tbgen_vitis_tcl)

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
