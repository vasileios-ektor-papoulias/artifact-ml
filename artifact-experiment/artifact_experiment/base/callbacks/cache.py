from abc import abstractmethod
from typing import Dict, Generic, List, Optional, TypeVar

from artifact_experiment.base.callbacks.base import (
    Callback,
    CallbackHandler,
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


CacheCallbackT = TypeVar("CacheCallbackT", bound=CacheCallback)


class CacheCallbackHandler(
    CallbackHandler[CacheCallbackT, CallbackResourcesTContr],
    Generic[CacheCallbackT, CallbackResourcesTContr, CacheDataT],
):
    def __init__(
        self,
        ls_callbacks: Optional[List[CacheCallbackT]] = None,
    ):
        super().__init__(ls_callbacks=ls_callbacks)
        self._cache: Dict[str, Optional[CacheDataT]] = {
            callback.key: callback.value for callback in self._ls_callbacks
        }

    @property
    def cache(self) -> Dict[str, Optional[CacheDataT]]:
        return self._cache

    @property
    def active_cache(self) -> Dict[str, CacheDataT]:
        active_cache = self._get_active_cache(cache=self._cache)
        return active_cache

    def execute(self, resources: CallbackResourcesTContr):
        self._clear()
        self._execute(resources=resources)
        self._update_cache()

    def update_cache(self):
        self._update_cache()

    def clear(self):
        self._clear()

    def _execute(self, resources: CallbackResourcesTContr):
        super().execute(resources=resources)

    def _update_cache(self):
        self._cache = {}
        for callback in self._ls_callbacks:
            self._cache.update(callback.cache)

    def _clear(self):
        for callback in self._ls_callbacks:
            callback.clear()
        self.update_cache()

    @staticmethod
    def _get_active_cache(
        cache: Dict[str, Optional[CacheDataT]],
    ) -> Dict[str, CacheDataT]:
        active_cache = {key: value for key, value in cache.items() if value is not None}
        return active_cache
