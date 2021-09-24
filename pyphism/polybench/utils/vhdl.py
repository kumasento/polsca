"""Dealing with the VHDL design files."""

import os
from dataclasses import dataclass
from typing import List, Tuple

from pyphism.utils import helper


@dataclass(eq=True, frozen=True)
class Port:
    """VHDL port definition."""

    name: str
    direction: str  # IN or OUT
    data_type: str


def parse_port_definition(line: str) -> Port:
    """Parse port definition into a Port object. Raise error if failed to parse."""
    line = line.strip()
    assert line.endswith(";")

    # Trim the last character ';', including the attached ' )' (note the whitespace)
    line = line[:-3] if line.endswith(" );") else line[:-1]

    # <NAME> : <IN or OUT> <DATA_TYPE>
    comps = line.split(":")
    assert len(comps) == 2

    # Parse name
    name = comps[0].strip()
    assert "," not in name, "Multiple ports in the same line is not supported."

    # Parse direction.
    dir_and_type = comps[1].strip()
    dir_and_type_comps = list(
        filter(lambda x: x, dir_and_type.split(" "))
    )  # filter empty
    assert len(dir_and_type_comps) >= 2

    dir = dir_and_type_comps[0].strip()
    assert dir in ("IN", "OUT"), f"{dir} is not IN or OUT"

    # Parse type
    data_type = " ".join(dir_and_type_comps[1:])

    return Port(name=name, direction=dir, data_type=data_type)


def get_port_start_and_end_pos(
    lines: str, name: str, is_component: bool = False
) -> Tuple[int, int]:
    """Get the start and end positions in the provided lines list that enclose all the ports.

    Note that this function will raise error if no port list can be found.
    """
    start_pattern = f"entity {name} is" if not is_component else f"component {name} is"

    # TODO: it is also possible to end with `end <name>;`.
    end_pattern = "end;" if not is_component else "end component;"

    start_pos = helper.find_substr_in_list(start_pattern, lines)
    assert start_pos >= 0 and start_pos < len(lines)
    end_pos = helper.find_substr_in_list(end_pattern, lines, start_pos)
    assert end_pos >= 0 and end_pos < len(lines)

    # +2 to ignore the `entity/component <name> is' and the `port (' lines.
    return start_pos + 2, end_pos


def get_port_list(lines: str, name: str, is_component: bool = False) -> List[str]:
    """Find the list of ports from the lines read from file.

    Note that this function will raise error if no port list can be found.
    """
    start_pos, end_pos = get_port_start_and_end_pos(
        lines, name, is_component=is_component
    )

    port_lines = lines[start_pos:end_pos]
    return list(map(parse_port_definition, port_lines))


def migrate_port_list(
    from_lines: List[str], to_lines: List[str], top_func: str
) -> List[str]:
    """Blend in the ports from `from_lines` into `to_lines`."""
    to_port_start_pos, to_port_end_pos = get_port_start_and_end_pos(to_lines, top_func)
    from_port_start_pos, from_port_end_pos = get_port_start_and_end_pos(
        from_lines, top_func, is_component=True
    )
    return (
        to_lines[:to_port_start_pos]
        + from_lines[from_port_start_pos:from_port_end_pos]
        + to_lines[to_port_end_pos:]
    )


def update_source_by_testbench(
    src_file: str, tbs_file: str, dst_file: str, top_func: str
):
    """
    Update the source file interface by the tbs file using -
    1. The port list of the top_func should be the same as the testbench.
    2. Reset the newly introduced write-enable signals (_we)

    The result will be populated to the dst_file.

    Args:
        src_file: the name of the original design file in vhdl
        tbs_file: the name of the test bench source
        dst_file: the name of the updated design file
        top_func: the name of the top-level design
    """
    assert os.path.isfile(src_file)
    assert os.path.isfile(tbs_file)
    assert dst_file != src_file and dst_file != tbs_file
    assert top_func
    assert src_file.endswith(".vhd") or src_file.endswith(".vhdl")
    assert tbs_file.endswith(".vhd") or tbs_file.endswith(".vhdl")

    # ------------------------ File read:
    src_lines = helper.read_lines_from_file(src_file)
    tbs_lines = helper.read_lines_from_file(tbs_file)

    # ------------------------ Sanity check:
    src_ports = get_port_list(src_lines, top_func)
    tbs_ports = get_port_list(tbs_lines, top_func, is_component=True)
    # Ports from the design file should be a subset of those in the testbench.
    assert set(src_ports).issubset(tbs_ports)

    # ------------------------ Step 1:
    # Blend in the ports from testbench into the source.
    dst_lines = migrate_port_list(tbs_lines, src_lines, top_func)

    # ------------------------ Step 2:
    # Newly introduced write-enable wires
    nwe_ports = [
        port for port in tbs_ports if port not in src_lines and "_we" in port.name
    ]
    # Find the first begin.
    begin_pos = helper.find_substr_in_list("begin", dst_lines)
    dst_lines = (
        dst_lines[: begin_pos + 1]
        # Reset write-enables
        + [f"    {port.name} <= '0';\n" for port in nwe_ports]
        + dst_lines[begin_pos + 1 :]
    )
    print(dst_lines)

    # ------------------------ Step 3:
    # Update the destination file.
    with open(dst_file, "w") as f:
        f.writelines(dst_lines)
