from typing import Any, Dict, Generic, Optional, Sequence, TypeVar

from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.callbacks.cache import CacheCallback
from artifact_experiment.base.handlers.base import CallbackHandler

CacheCallbackTCov = TypeVar("CacheCallbackTCov", bound=CacheCallback[Any, Any], covariant=True)
CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", covariant=True)


class CacheCallbackHandler(
    CallbackHandler[CacheCallbackTCov, CallbackResourcesTContr],
    Generic[CacheCallbackTCov, CallbackResourcesTContr, CacheDataTCov],
):
    def __init__(self, callbacks: Optional[Sequence[CacheCallbackTCov]] = None):
        super().__init__(callbacks=callbacks)
        self._cache: Dict[str, Optional[CacheDataTCov]] = {
            callback.key: callback.value for callback in self._ls_callbacks
        }

    @property
    def cache(self) -> Dict[str, Optional[CacheDataTCov]]:
        return self._cache

    @property
    def active_cache(self) -> Dict[str, CacheDataTCov]:
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
        cache: Dict[str, Optional[CacheDataTCov]],
    ) -> Dict[str, CacheDataTCov]:
        active_cache = {key: value for key, value in cache.items() if value is not None}
        return active_cache
