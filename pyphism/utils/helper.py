""" Helper functions. """
import functools
import json
import logging
import os
import pprint
import subprocess
import time
from typing import List, Optional

import colorlog

# ------------------------- Operating files --------------------


def prepend_to_file(path: str, text: str):
    assert os.path.isfile(path)

    # First read from the file
    with open(path, "r") as f:
        data = f.read()
    # then modify it
    with open(path, "w") as f:
        f.write(text + "\n" + data)


def read_lines_from_file(path: str, strip: bool = False) -> List[str]:
    """Read file into stripped lines."""
    with open(path, "r") as f:
        lines = f.readlines()

    if not strip:
        return lines

    return list([line.strip() for line in lines])


def is_cosim_setup(file: str) -> bool:
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


# ------------------------ List helpers -----------------------


def find_substr_in_list(
    substr: str, strs: List[str], start_pos: int = 0, exact: bool = False
) -> int:
    """Find a specific substring within the given list of strings,
    starting from a given position."""
    for i, s in enumerate(strs):
        if i < start_pos:
            continue
        # Exact match
        if exact and substr == s:
            return i
        # Contain
        if not exact and substr in s:
            return i

    return -1


# -------------------------- Date and time -------------------------


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


# -------------------------- Analysis --------------------------------


def get_param_names(func_name: str, src_file: str, clang_path: str):
    """From the given C file, we try to extract the top function's parameter list.
    This will be useful for Vitis LLVM rewrite."""

    def is_func_decl(item, name):
        return item["kind"] == "FunctionDecl" and item["name"] == name

    # Get the corresponding AST in JSON.
    proc = subprocess.Popen(
        [
            clang_path,
            src_file,
            "-Xclang",
            "-ast-dump=json",
            "-fsyntax-only",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    data = json.loads(proc.stdout.read())

    # First find the top function declaration entry.
    func_name_decls = list(
        filter(functools.partial(is_func_decl, name=func_name), data["inner"])
    )
    assert (
        len(func_name_decls) >= 1
    ), "Should have at least a single declaration for the provided function."
    func_name_decl = func_name_decls[0]

    # Then get all ParmVarDecl.
    parm_var_decls = filter(
        lambda x: x["kind"] == "ParmVarDecl", func_name_decl["inner"]
    )
    return [decl["name"] for decl in parm_var_decls]


# -------------------------- Others --------------------------------


def get_project_root():
    """Get the root directory of the project."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_logger(name: str, log_file: str = "", console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if log_file:
        if os.path.isfile(log_file):
            os.remove(log_file)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    # Console handler
    if console:
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
        logger.addHandler(ch)

    return logger
