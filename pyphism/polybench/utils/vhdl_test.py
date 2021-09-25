import os

from pyphism.polybench.utils import vhdl


def test_parse_port_definition():
    port = vhdl.parse_port_definition("ap_clk : IN STD_LOGIC;")
    assert port.name == "ap_clk"
    assert port.direction == "IN"
    assert port.data_type == "STD_LOGIC"

    port = vhdl.parse_port_definition("nl : IN STD_LOGIC_VECTOR (31 downto 0);")
    assert port.name == "nl"
    assert port.direction == "IN"
    assert port.data_type == "STD_LOGIC_VECTOR (31 DOWNTO 0)"

    port = vhdl.parse_port_definition("D_5_d1 : OUT STD_LOGIC_VECTOR (63 downto 0) );")
    assert port.name == "D_5_d1"
    assert port.direction == "OUT"
    assert port.data_type == "STD_LOGIC_VECTOR (63 DOWNTO 0)"


def test_get_port_list_on_entity():
    lines = [
        "entity top is",
        "port (",
        "ap_clk : IN STD_LOGIC;",
        "ap_rst : IN STD_LOGIC;",
        "ap_start : IN STD_LOGIC);",
        "end;",
    ]
    ports = vhdl.get_port_list(lines, "top", is_component=False)
    assert len(ports) == 3


def test_get_port_list_on_component():
    lines = [
        "component top is",
        "port (",
        "ap_clk : IN STD_LOGIC;",
        "ap_rst : IN STD_LOGIC;",
        "ap_start : IN STD_LOGIC);",
        "end component;",
    ]
    ports = vhdl.get_port_list(lines, "top", is_component=True)
    assert len(ports) == 3


def test_subset():
    ports = [
        vhdl.Port("ap_clk", "IN", "STD_LOGIC"),
        vhdl.Port("ap_rst", "IN", "STD_LOGIC"),
        vhdl.Port("ap_start", "IN", "STD_LOGIC"),
    ]
    assert set(ports[1:]).issubset(ports)


def test_migrate_port_list():
    to_lines = [
        "entity top is",
        "port (",
        "ap_clk : IN STD_LOGIC;",
        "ap_start : IN STD_LOGIC);",
        "end;",
    ]
    from_lines = [
        "component top is",
        "port (",
        "ap_clk : IN STD_LOGIC;",
        "ap_rst : IN STD_LOGIC;",
        "ap_start : IN STD_LOGIC);",
        "end component;",
    ]
    assert vhdl.migrate_port_list(from_lines, to_lines, "top") == [
        "entity top is",
        "port (",
        "ap_clk : IN STD_LOGIC;",
        "ap_rst : IN STD_LOGIC;",
        "ap_start : IN STD_LOGIC);",
        "end;",
    ]


TEST_SRC_FILE = """
entity top is
port (
    ap_clk : IN STD_LOGIC;
    ni : IN STD_LOGIC_VECTOR (31 downto 0);
    A_address0 : OUT STD_LOGIC_VECTOR (9 downto 0);
    A_ce0 : OUT STD_LOGIC;
    A_q0 : IN STD_LOGIC_VECTOR (63 downto 0);
    A_address1 : OUT STD_LOGIC_VECTOR (9 downto 0);
    A_ce1 : OUT STD_LOGIC;
    A_d1 : OUT STD_LOGIC_VECTOR (63 downto 0) );
end;

begin
"""

TEST_TBS_FILE = """
component top is
port (
    ap_clk : IN STD_LOGIC;
    ni : IN STD_LOGIC_VECTOR (31 downto 0);
    A_address0 : OUT STD_LOGIC_VECTOR (9 downto 0);
    A_ce0 : OUT STD_LOGIC;
    A_q0 : IN STD_LOGIC_VECTOR (63 downto 0);
    A_address1 : OUT STD_LOGIC_VECTOR (9 downto 0);
    A_ce1 : OUT STD_LOGIC;
    A_we1 : OUT STD_LOGIC;
    A_d1 : OUT STD_LOGIC_VECTOR (63 downto 0) );
end component;
"""

TEST_DST_FILE = """
entity top is
port (
    ap_clk : IN STD_LOGIC;
    ni : IN STD_LOGIC_VECTOR (31 downto 0);
    A_address0 : OUT STD_LOGIC_VECTOR (9 downto 0);
    A_ce0 : OUT STD_LOGIC;
    A_q0 : IN STD_LOGIC_VECTOR (63 downto 0);
    A_address1 : OUT STD_LOGIC_VECTOR (9 downto 0);
    A_ce1 : OUT STD_LOGIC;
    A_we1 : OUT STD_LOGIC;
    A_d1 : OUT STD_LOGIC_VECTOR (63 downto 0) );
end;

begin
    A_we1 <= '0';
"""


def test_update_source_by_testbench(tmp_path):
    src_file = os.path.join(tmp_path, "src.vhd")
    tbs_file = os.path.join(tmp_path, "tbs.vhd")
    with open(src_file, "w") as f:
        f.write(TEST_SRC_FILE)
    with open(tbs_file, "w") as f:
        f.write(TEST_TBS_FILE)

    dst_file = os.path.join(tmp_path, "dst.vhd")

    vhdl.update_source_by_testbench(src_file, tbs_file, dst_file, top_func="top")

    assert os.path.isfile(dst_file), "File should be created."
    with open(dst_file, "r") as f:
        lines = f.readlines()
        assert "".join(lines) == TEST_DST_FILE, "Port list should be updated."
