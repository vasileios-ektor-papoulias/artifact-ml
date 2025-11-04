from abc import abstractmethod
from typing import Dict, Generic, Optional, TypeVar

from artifact_experiment.base.callbacks.base import (
    Callback,
    CallbackResources,
)

CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", covariant=True)


class CacheCallback(
    Callback[CallbackResourcesTContr], Generic[CallbackResourcesTContr, CacheDataTCov]
):
    def __init__(self, key: str):
        self._key = key
        self._cache: Dict[str, Optional[CacheDataTCov]] = {self._key: None}

    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> CacheDataTCov: ...

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> Optional[CacheDataTCov]:
        return next(iter(self._cache.values()), None)

    @property
    def cache(self) -> Dict[str, Optional[CacheDataTCov]]:
        return self._cache.copy()

    def execute(self, resources: CallbackResourcesTContr):
        self._clear()
        value = self._compute(resources=resources)
        self._cache[self._key] = value

    def clear(self):
        self._clear()

    def _clear(self):
        self._cache = {self._key: None}
