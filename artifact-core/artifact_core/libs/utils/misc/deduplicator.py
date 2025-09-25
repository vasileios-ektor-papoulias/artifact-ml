import warnings
from collections.abc import Hashable
from typing import List, Sequence, Set, TypeVar

T = TypeVar("T", bound=Hashable)


class Deduplicator:
    @staticmethod
    def deduplicate(seq: Sequence[T], warn: bool = True) -> List[T]:
        seen: Set[T] = set()
        out: List[T] = []
        dups: List[T] = []
        for x in seq:
            if x in seen:
                if x not in dups:
                    dups.append(x)
            else:
                seen.add(x)
                out.append(x)
        if warn and dups:
            warnings.warn(f"Input contained duplicates (removed): {dups}")
        return out

    @staticmethod
    def to_set(seq: Sequence[T], warn: bool = True) -> Set[T]:
        return set(Deduplicator.deduplicate(seq=seq, warn=warn))
