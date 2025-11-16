import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Generic, TypeVar

StoreDataT = TypeVar("StoreDataT")


class ClearMLStore(ABC, Generic[StoreDataT]):
    def __init__(self, root_dir: str):
        self._root_dir = root_dir

    @property
    def root_dir(self) -> str:
        return deepcopy(self._root_dir)

    @property
    @abstractmethod
    def n_entries(self) -> int: ...

    @abstractmethod
    def _get(self, path: str) -> StoreDataT: ...

    def __len__(self) -> int:
        return self.n_entries

    def get(self, path: str) -> StoreDataT:
        path = self._prepend_root_dir(path=path, root_dir=self._root_dir)
        data = self._get(path=path)
        return data

    @staticmethod
    def _prepend_root_dir(path: str, root_dir: str) -> str:
        if not path.startswith(root_dir):
            path = os.path.join(root_dir, path)
        return path
