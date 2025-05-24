from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from artifact_experiment.base.callbacks.base import (
    CallbackResources,
)
from artifact_experiment.base.callbacks.cache import (
    CacheCallback,
    CacheCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

CacheDataT = TypeVar("CacheDataT")
CallbackResourcesT = TypeVar("CallbackResourcesT", bound=CallbackResources)
CacheCallbackT = TypeVar(
    "CacheCallbackT",
    bound=CacheCallback,
)
ScoreCallbackT = TypeVar("ScoreCallbackT", bound=CacheCallback[Any, float])
ArrayCallbackT = TypeVar("ArrayCallbackT", bound=CacheCallback[Any, ndarray])
PlotCallbackT = TypeVar("PlotCallbackT", bound=CacheCallback[Any, Figure])
ScoreCollectionCallbackT = TypeVar(
    "ScoreCollectionCallbackT", bound=CacheCallback[Any, Dict[str, float]]
)
ArrayCollectionCallbackT = TypeVar(
    "ArrayCollectionCallbackT", bound=CacheCallback[Any, Dict[str, ndarray]]
)
PlotCollectionCallbackT = TypeVar(
    "PlotCollectionCallbackT", bound=CacheCallback[Any, Dict[str, Figure]]
)


class TrackingCallback(
    CacheCallback[CallbackResourcesT, CacheDataT],
    Generic[CallbackResourcesT, CacheDataT],
):
    def __init__(self, key: str, tracking_client: Optional[TrackingClient] = None):
        super().__init__(key=key)
        self._tracking_client = tracking_client

    @abstractmethod
    def _compute(self, resources: CallbackResourcesT) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def execute(self, resources: CallbackResourcesT):
        super().execute(resources=resources)
        if self._tracking_client is not None:
            assert self.value is not None
            self._export(key=self.key, value=self.value, tracking_client=self._tracking_client)


class ScoreTrackingCallback(
    TrackingCallback[CallbackResourcesT, float], Generic[CallbackResourcesT]
):
    @staticmethod
    def _export(key: str, value: float, tracking_client: TrackingClient):
        tracking_client.log_score(score=value, name=key)


class ArrayTrackingCallback(
    TrackingCallback[CallbackResourcesT, ndarray], Generic[CallbackResourcesT]
):
    @staticmethod
    def _export(key: str, value: ndarray, tracking_client: TrackingClient):
        tracking_client.log_array(array=value, name=key)


class PlotTrackingCallback(
    TrackingCallback[CallbackResourcesT, Figure], Generic[CallbackResourcesT]
):
    @staticmethod
    def _export(key: str, value: Figure, tracking_client: TrackingClient):
        tracking_client.log_plot(plot=value, name=key)


class ScoreCollectionTrackingCallback(
    TrackingCallback[CallbackResourcesT, Dict[str, float]], Generic[CallbackResourcesT]
):
    @staticmethod
    def _export(key: str, value: Dict[str, float], tracking_client: TrackingClient):
        tracking_client.log_score_collection(score_collection=value, name=key)


class ArrayCollectionTrackingCallback(
    TrackingCallback[CallbackResourcesT, Dict[str, ndarray]], Generic[CallbackResourcesT]
):
    @staticmethod
    def _export(key: str, value: Dict[str, ndarray], tracking_client: TrackingClient):
        tracking_client.log_array_collection(array_collection=value, name=key)


class PlotCollectionTrackingCallback(
    TrackingCallback[CallbackResourcesT, Dict[str, Figure]], Generic[CallbackResourcesT]
):
    @staticmethod
    def _export(key: str, value: Dict[str, Figure], tracking_client: TrackingClient):
        tracking_client.log_plot_collection(plot_collection=value, name=key)


class TrackingCallbackHandler(
    CacheCallbackHandler[CacheCallbackT, CallbackResourcesT, CacheDataT],
    Generic[CacheCallbackT, CallbackResourcesT, CacheDataT],
):
    def __init__(
        self,
        ls_callbacks: Optional[List[CacheCallbackT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(ls_callbacks=ls_callbacks)
        self._tracking_client = tracking_client

    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient):
        pass

    def execute(self, resources: CallbackResourcesT):
        super().execute(resources=resources)
        if self._tracking_client is not None:
            self._export(
                cache=self.active_cache,
                tracking_client=self._tracking_client,
            )


class ScoreCallbackHandler(
    TrackingCallbackHandler[ScoreCallbackT, CallbackResourcesT, float],
    Generic[ScoreCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(cache: Dict[str, float], tracking_client: TrackingClient):
        for score_name, score_value in cache.items():
            tracking_client.log_score(score=score_value, name=score_name)


class ArrayCallbackHandler(
    TrackingCallbackHandler[ArrayCallbackT, CallbackResourcesT, ndarray],
    Generic[ArrayCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(cache: Dict[str, ndarray], tracking_client: TrackingClient):
        for array_name, array in cache.items():
            tracking_client.log_array(array=array, name=array_name)


class PlotCallbackHandler(
    TrackingCallbackHandler[PlotCallbackT, CallbackResourcesT, Figure],
    Generic[PlotCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(
        cache: Dict[str, Figure],
        tracking_client: TrackingClient,
    ):
        for plot_name, plot in cache.items():
            tracking_client.log_plot(plot=plot, name=plot_name)


class ScoreCollectionCallbackHandler(
    TrackingCallbackHandler[
        ScoreCollectionCallbackT,
        CallbackResourcesT,
        Dict[str, float],
    ],
    Generic[ScoreCollectionCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(
        cache: Dict[str, Dict[str, float]],
        tracking_client: TrackingClient,
    ):
        for score_collection_name, score_collection in cache.items():
            tracking_client.log_score_collection(
                score_collection=score_collection, name=score_collection_name
            )


class ArrayCollectionCallbackHandler(
    TrackingCallbackHandler[
        ArrayCollectionCallbackT,
        CallbackResourcesT,
        Dict[str, ndarray],
    ],
    Generic[ArrayCollectionCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(
        cache: Dict[str, Dict[str, ndarray]],
        tracking_client: TrackingClient,
    ):
        for array_collection_name, array_collection in cache.items():
            tracking_client.log_array_collection(
                array_collection=array_collection, name=array_collection_name
            )


class PlotCollectionCallbackHandler(
    TrackingCallbackHandler[
        PlotCollectionCallbackT,
        CallbackResourcesT,
        Dict[str, Figure],
    ],
    Generic[PlotCollectionCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(
        cache: Dict[str, Dict[str, Figure]],
        tracking_client: TrackingClient,
    ):
        for plot_collection_name, plot_collection in cache.items():
            tracking_client.log_plot_collection(
                plot_collection=plot_collection, name=plot_collection_name
            )
