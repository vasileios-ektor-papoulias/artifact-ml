from typing import List, Mapping

import pytest
from artifact_core._utils.collections.map_aligner import MapAligner


@pytest.mark.unit
@pytest.mark.parametrize(
    "left, right, expected_keys, expected_vals_left, expected_vals_right",
    [
        (
            {"a": 1, "b": 2},
            {"a": 10, "b": 20},
            ["a", "b"],
            [1, 2],
            [10, 20],
        ),
        (
            {"x": "val1"},
            {"x": "val2"},
            ["x"],
            ["val1"],
            ["val2"],
        ),
        (
            {},
            {},
            [],
            [],
            [],
        ),
        (
            {"key1": 100, "key2": 200, "key3": 300},
            {"key1": 1000, "key2": 2000, "key3": 3000},
            ["key1", "key2", "key3"],
            [100, 200, 300],
            [1000, 2000, 3000],
        ),
        # Extra keys in right are allowed
        (
            {"a": 1, "b": 2},
            {"a": 10, "b": 20, "c": 30, "d": 40},
            ["a", "b"],
            [1, 2],
            [10, 20],
        ),
        # Preserve left key order
        (
            {"z": 1, "a": 2, "m": 3},
            {"m": 30, "z": 10, "a": 20},
            ["z", "a", "m"],
            [1, 2, 3],
            [10, 20, 30],
        ),
        # Different key types
        (
            {1: "a", 2: "b"},
            {1: "x", 2: "y"},
            [1, 2],
            ["a", "b"],
            ["x", "y"],
        ),
        # None values
        (
            {"a": None, "b": 2},
            {"a": 10, "b": None},
            ["a", "b"],
            [None, 2],
            [10, None],
        ),
    ],
)
def test_align(
    left: Mapping,
    right: Mapping,
    expected_keys: List,
    expected_vals_left: List,
    expected_vals_right: List,
):
    keys, vals_left, vals_right = MapAligner.align(left=left, right=right)

    assert keys == expected_keys
    assert vals_left == expected_vals_left
    assert vals_right == expected_vals_right


@pytest.mark.unit
@pytest.mark.parametrize(
    "left, right",
    [
        ({"a": 1, "b": 2}, {"a": 10}),
        ({"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20}),
        ({"x": 1}, {}),
    ],
)
def test_align_missing_keys_raises(left: Mapping, right: Mapping):
    with pytest.raises(KeyError, match="Right mapping missing .* id\\(s\\)"):
        MapAligner.align(left=left, right=right)
