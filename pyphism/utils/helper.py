""" Helper functions. """

import functools
import json
import os
import subprocess
import time
from typing import List, Optional

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
        len(func_name_decls) == 1
    ), "Should be a single declaration for the provided function."
    func_name_decl = func_name_decls[0]

    # Then get all ParmVarDecl.
    parm_var_decls = filter(
        lambda x: x["kind"] == "ParmVarDecl", func_name_decl["inner"]
    )
    return [decl["name"] for decl in parm_var_decls]
