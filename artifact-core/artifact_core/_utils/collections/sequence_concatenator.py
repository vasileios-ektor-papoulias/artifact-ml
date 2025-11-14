from itertools import chain
from typing import List, Sequence, TypeVar

T = TypeVar("T")


class SequenceConcatenator:
    @staticmethod
    def concatenate(seq1: Sequence[T], seq2: Sequence[T], /) -> List[T]:
        return list(chain(seq1, seq2))
