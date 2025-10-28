from typing import Any, Hashable, Mapping, Set, TypeVar

HashableT = TypeVar("HashableT", bound=Hashable)


class KeySelector:
    @staticmethod
    def get_common_keys(*dicts: Mapping[HashableT, Any]) -> Set[HashableT]:
        return set(dicts[0]).intersection(*dicts[1:]) if dicts else set()
