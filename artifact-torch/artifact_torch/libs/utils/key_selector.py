from typing import Any, Hashable, Mapping, Set, TypeVar

HashableT = TypeVar("HashableT", bound=Hashable)


class KeySelector:
    @staticmethod
    def get_non_null_keys(d: Mapping[HashableT, Any]) -> Set[HashableT]:
        return {k for k, v in d.items() if v is not None}

    @staticmethod
    def get_common_non_null_keys(*dicts: Mapping[HashableT, Any]) -> Set[HashableT]:
        if not dicts:
            return set()
        common = set(dicts[0]).intersection(*dicts[1:])
        return {k for k in common if all(d.get(k) is not None for d in dicts)}

    @staticmethod
    def get_common_keys(*dicts: Mapping[HashableT, Any]) -> Set[HashableT]:
        return set(dicts[0]).intersection(*dicts[1:]) if dicts else set()
