from typing import Any, Dict, Generic, Mapping, Optional, Sequence, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.components.callbacks.cache import (
    CacheCallback,
    CacheCallbackResources,
)
from artifact_experiment._base.components.handlers.base import CallbackHandler
from artifact_experiment._base.typing.tracking_data import TrackingData

CacheCallbackTCov = TypeVar("CacheCallbackTCov", bound=CacheCallback[Any, Any], covariant=True)
CacheCallbackResourcesTContr = TypeVar(
    "CacheCallbackResourcesTContr", bound=CacheCallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", bound=TrackingData, covariant=True)


class CacheCallbackHandler(
    CallbackHandler[CacheCallbackTCov, CacheCallbackResourcesTContr],
    Generic[CacheCallbackTCov, CacheCallbackResourcesTContr, CacheDataTCov],
):
    def __init__(self, callbacks: Optional[Sequence[CacheCallbackTCov]] = None):
        super().__init__(callbacks=callbacks)
        self._cache: Dict[str, Optional[CacheDataTCov]] = {
            callback.key: callback.value for callback in self._ls_callbacks
        }

    @property
    def cache(self) -> Mapping[str, Optional[CacheDataTCov]]:
        return self._cache

    @property
    def active_cache(self) -> Mapping[str, CacheDataTCov]:
        active_cache = self._get_active_cache(cache=self._cache)
        return active_cache

    def execute(self, resources: CacheCallbackResourcesTContr):
        self._clear()
        self._execute(resources=resources)
        self._update_cache()

    def update_cache(self):
        self._update_cache()

    def clear(self):
        self._clear()

    def _execute(self, resources: CacheCallbackResourcesTContr):
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


CacheScoreHandler = CacheCallbackHandler[CacheCallbackTCov, CacheCallbackResourcesTContr, Score]
CacheArrayHandler = CacheCallbackHandler[CacheCallbackTCov, CacheCallbackResourcesTContr, Array]
CachePlotHandler = CacheCallbackHandler[CacheCallbackTCov, CacheCallbackResourcesTContr, Plot]
CacheScoreCollectionHandler = CacheCallbackHandler[
    CacheCallbackTCov, CacheCallbackResourcesTContr, ScoreCollection
]
CacheArrayCollectionHandler = CacheCallbackHandler[
    CacheCallbackTCov, CacheCallbackResourcesTContr, ArrayCollection
]
CachePlotCollectionHandler = CacheCallbackHandler[
    CacheCallbackTCov, CacheCallbackResourcesTContr, PlotCollection
]
