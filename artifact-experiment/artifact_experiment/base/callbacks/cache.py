from abc import abstractmethod
from typing import Dict, Generic, Optional, TypeVar

from artifact_experiment.base.callbacks.base import (
    Callback,
    CallbackResources,
)

CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)
CacheDataT = TypeVar("CacheDataT")


class CacheCallback(
    Callback[CallbackResourcesTContr], Generic[CallbackResourcesTContr, CacheDataT]
):
    def __init__(self, key: str):
        self._key = key
        self._cache: Dict[str, Optional[CacheDataT]] = {self._key: None}

    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> CacheDataT: ...

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> Optional[CacheDataT]:
        return next(iter(self._cache.values()), None)

    @property
    def cache(self) -> Dict[str, Optional[CacheDataT]]:
        return self._cache.copy()

    def execute(self, resources: CallbackResourcesTContr):
        self._clear()
        value = self._compute(resources=resources)
        self._cache[self._key] = value

    def clear(self):
        self._clear()

    def _clear(self):
        self._cache = {self._key: None}
