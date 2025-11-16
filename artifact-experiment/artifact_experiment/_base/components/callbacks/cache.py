from abc import abstractmethod
from typing import Dict, Generic, Mapping, Optional, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.components.callbacks.base import Callback
from artifact_experiment._base.components.resources.cache import CacheCallbackResources
from artifact_experiment._base.typing.tracking_data import TrackingData

CacheCallbackResourcesTContr = TypeVar(
    "CacheCallbackResourcesTContr", bound=CacheCallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", bound=TrackingData, covariant=True)


class CacheCallback(
    Callback[CacheCallbackResourcesTContr], Generic[CacheCallbackResourcesTContr, CacheDataTCov]
):
    def __init__(self, base_key: str):
        self._base_key = base_key
        self._cache: Dict[str, Optional[CacheDataTCov]] = {self._base_key: None}

    @abstractmethod
    def _compute(self, resources: CacheCallbackResourcesTContr) -> CacheDataTCov: ...

    @property
    def base_key(self) -> str:
        return self._base_key

    @property
    def key(self) -> str:
        return next(iter(self._cache.keys()), self._base_key)

    @property
    def value(self) -> Optional[CacheDataTCov]:
        return next(iter(self._cache.values()), None)

    @property
    def cache(self) -> Mapping[str, Optional[CacheDataTCov]]:
        return self._cache.copy()

    def execute(self, resources: CacheCallbackResourcesTContr):
        self._clear()
        qualified_key = self._qualify_base_key(base_key=self._base_key, resources=resources)
        self._cache[qualified_key] = None
        value = self._compute(resources=resources)
        self._cache[qualified_key] = value

    def clear(self):
        self._cache = {self._base_key: None}

    def _clear(self):
        self._cache = {}

    @classmethod
    def _qualify_base_key(cls, base_key: str, resources: CacheCallbackResourcesTContr) -> str:
        key = base_key
        if resources.trigger is not None:
            key = f"{key}_{resources.trigger}"
        return key


CacheScoreCallback = CacheCallback[CacheCallbackResourcesTContr, Score]
CacheArrayCallback = CacheCallback[CacheCallbackResourcesTContr, Array]
CachePlotCallback = CacheCallback[CacheCallbackResourcesTContr, Plot]
CacheScoreCollectionCallback = CacheCallback[CacheCallbackResourcesTContr, ScoreCollection]
CacheArrayCollectionCallback = CacheCallback[CacheCallbackResourcesTContr, ArrayCollection]
CachePlotCollectionCallback = CacheCallback[CacheCallbackResourcesTContr, PlotCollection]
