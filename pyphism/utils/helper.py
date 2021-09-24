""" Helper functions. """

import os
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
