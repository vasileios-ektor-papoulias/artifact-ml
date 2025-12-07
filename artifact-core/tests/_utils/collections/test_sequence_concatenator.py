from typing import List, Sequence

import pytest
from artifact_core._utils.collections.sequence_concatenator import SequenceConcatenator


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq1, seq2, expected",
    [
        ([1, 2], [3, 4], [1, 2, 3, 4]),
        ([1], [2], [1, 2]),
        ([], [1, 2], [1, 2]),
        ([1, 2], [], [1, 2]),
        ([], [], []),
        ([1, 2, 3], [4, 5, 6], [1, 2, 3, 4, 5, 6]),
        (["a", "b"], ["c", "d"], ["a", "b", "c", "d"]),
        (["hello"], ["world"], ["hello", "world"]),
        ([(1, 2), (3, 4)], [(5, 6)], [(1, 2), (3, 4), (5, 6)]),
        # Duplicates preserved
        ([1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3]),
        # Mixed types
        ([1, "a", 3.14], [True, None], [1, "a", 3.14, True, None]),
        # Different sequence types
        ((1, 2), [3, 4], [1, 2, 3, 4]),
        ([1, 2], (3, 4), [1, 2, 3, 4]),
        # Order preserved
        ([3, 1, 4], [1, 5, 9], [3, 1, 4, 1, 5, 9]),
    ],
)
def test_concatenate(seq1: Sequence, seq2: Sequence, expected: List):
    result = SequenceConcatenator.concatenate(seq1, seq2)
    assert result == expected
    assert isinstance(result, list)


@pytest.mark.unit
def test_concatenate_positional_only():
    result = SequenceConcatenator.concatenate([1], [2])
    assert result == [1, 2]
    with pytest.raises(TypeError):
        SequenceConcatenator.concatenate(seq1=[1], seq2=[2])
