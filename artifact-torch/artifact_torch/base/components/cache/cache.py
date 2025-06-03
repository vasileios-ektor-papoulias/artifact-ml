from collections import defaultdict
from enum import Enum
from typing import Dict, Generic, List, Optional, TypeVar


class CacheMode(Enum):
    STANDARD = "STANDARD"
    ALIGNED = "ALIGNED"


CacheDataT = TypeVar("CacheDataT")


class Cache(Generic[CacheDataT]):
    def __init__(self, mode: CacheMode = CacheMode.STANDARD):
        self._mode = mode
        self._cache: Dict[str, List[Optional[CacheDataT]]] = defaultdict(list)

    @property
    def keys(self) -> List[str]:
        return list(self._cache.keys())

    @property
    def n_entries(self) -> int:
        if not self._cache:
            return 0
        first_key = next(iter(self._cache))
        return len(self._cache[first_key])

    @property
    def is_empty(self) -> bool:
        return self.n_entries == 0

    def __getitem__(self, key: str) -> Optional[List[Optional[CacheDataT]]]:
        return self._cache.get(key)

    def get_full_history(self, key: str) -> Optional[List[Optional[CacheDataT]]]:
        return self._cache[key]

    def get_latest(self, key: str, default: Optional[CacheDataT] = None) -> Optional[CacheDataT]:
        history = self._cache.get(key)
        if history is None:
            latest = default
        else:
            latest = history[-1]
        return latest

    def clear(self) -> None:
        self._cache.clear()

    def append(self, items: Dict[str, CacheDataT]):
        if self._mode == CacheMode.STANDARD:
            self._append_standard(items=items)
        elif self._mode == CacheMode.ALIGNED:
            self._append_aligned(items=items)
        else:
            raise ValueError(f"Invalid cache mode: {self._mode}")

    def append_standard(self, items: Dict[str, CacheDataT]):
        for key, value in items.items():
            self._cache[key].append(value)

    def _append_standard(self, items: Dict[str, CacheDataT]):
        for key, value in items.items():
            self._cache[key].append(value)

    def _append_aligned(self, items: Dict[str, CacheDataT]):
        ls_new_keys = [key for key in items.keys() if key not in self._cache.keys()]
        for key in ls_new_keys:
            self._add_empty_history(cache=self._cache, key=key, n_entries=self.n_entries)
        self._add_items_aligned(cache=self._cache, items=items)

    @staticmethod
    def _add_items_aligned(
        cache: Dict[str, List[Optional[CacheDataT]]], items: Dict[str, CacheDataT]
    ):
        for key, col_list in cache.items():
            if key in items:
                col_list.append(items[key])
            else:
                col_list.append(None)

    @staticmethod
    def _add_empty_history(cache: Dict[str, List[Optional[CacheDataT]]], key: str, n_entries: int):
        cache[key] = [None] * n_entries


class StandardCache(Cache[CacheDataT], Generic[CacheDataT]):
    def __init__(self):
        super().__init__(mode=CacheMode.STANDARD)


class AlignedCache(Cache[CacheDataT], Generic[CacheDataT]):
    def __init__(self):
        super().__init__(mode=CacheMode.ALIGNED)
