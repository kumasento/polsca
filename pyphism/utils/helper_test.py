import shutil
import tempfile

import pyphism.utils.helper as helper


def test_find_substr_in_list():
    strs = ["abcd", "abcd", "efg", "hijk"]
    assert helper.find_substr_in_list("abc", strs) == 0
    assert helper.find_substr_in_list("abc", strs, start_pos=1) == 1
    assert helper.find_substr_in_list("abc", strs, start_pos=2) == -1
    assert helper.find_substr_in_list("egi1", strs) == -1


def test_get_param_names():
    clang_path = shutil.which("clang")
    if not clang_path:
        return

    src_file = tempfile.NamedTemporaryFile(suffix=".c")
    with open(src_file.name, "w") as f:
        f.write("void foo(int a, float b[32]) {}")

    assert helper.get_param_names("foo", src_file.name, clang_path) == ["a", "b"]
