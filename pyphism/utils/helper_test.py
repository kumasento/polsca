import pyphism.utils.helper as helper


def test_find_substr_in_list():
    strs = ["abcd", "abcd", "efg", "hijk"]
    assert helper.find_substr_in_list("abc", strs) == 0
    assert helper.find_substr_in_list("abc", strs, start_pos=1) == 1
    assert helper.find_substr_in_list("abc", strs, start_pos=2) == -1
    assert helper.find_substr_in_list("egi1", strs) == -1
