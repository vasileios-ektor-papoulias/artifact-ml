from typing import List, Sequence, Set

import pytest
from artifact_core._utils.collections.deduplicator import Deduplicator


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq, expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 2, 3], [1, 2, 3]),
        ([1, 1, 1], [1]),
        ([1, 2, 3, 2, 1], [1, 2, 3]),
        ([], []),
        ([1], [1]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
        ([(1, 2), (3, 4), (1, 2)], [(1, 2), (3, 4)]),
        ([True, False, True], [True, False]),
        ([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], [3, 1, 4, 5, 9, 2, 6]),
    ],
)
def test_deduplicate(seq: Sequence, expected: List):
    result = Deduplicator.deduplicate(seq=seq, warn=False)
    assert result == expected
    assert isinstance(result, list)


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq, expected",
    [
        ([1, 2, 2, 3], [1, 2, 3]),
        ([1, 1, 1], [1]),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
    ],
)
def test_deduplicate_warns(seq: Sequence, expected: List):
    with pytest.warns(UserWarning, match="Input contained duplicates"):
        result = Deduplicator.deduplicate(seq=seq, warn=True)
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq, expected_set",
    [
        ([1, 2, 3], {1, 2, 3}),
        ([1, 2, 2, 3], {1, 2, 3}),
        ([1, 1, 1], {1}),
        ([], set()),
        (["a", "b", "c", "a"], {"a", "b", "c"}),
    ],
)
def test_to_set(seq: Sequence, expected_set: Set):
    result = Deduplicator.to_set(seq=seq, warn=False)
    assert result == expected_set
    assert isinstance(result, set)


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq, expected_set",
    [
        ([1, 2, 2, 3], {1, 2, 3}),
        (["a", "b", "a"], {"a", "b"}),
    ],
)
def test_to_set_warns(seq: Sequence, expected_set: Set):
    with pytest.warns(UserWarning, match="Input contained duplicates"):
        result = Deduplicator.to_set(seq=seq, warn=True)
    assert result == expected_set
