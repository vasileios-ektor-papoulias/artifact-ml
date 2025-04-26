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


class TrackingCallback(
    CacheCallback[CallbackResourcesT, CacheDataT],
    Generic[CallbackResourcesT, CacheDataT],
):
    @abstractmethod
    def _compute(self, resources: CallbackResourcesT) -> CacheDataT: ...


TrackingCallbackT = TypeVar(
    "TrackingCallbackT",
    bound=TrackingCallback,
)
scoreTrackingCallbackT = TypeVar("scoreTrackingCallbackT", bound=TrackingCallback[Any, float])
arrayTrackingCallbackT = TypeVar("arrayTrackingCallbackT", bound=TrackingCallback[Any, ndarray])
plotTrackingCallbackT = TypeVar("plotTrackingCallbackT", bound=TrackingCallback[Any, Figure])
scoreCollectionTrackingCallbackT = TypeVar(
    "scoreCollectionTrackingCallbackT", bound=TrackingCallback[Any, Dict[str, float]]
)
arrayCollectionTrackingCallbackT = TypeVar(
    "arrayCollectionTrackingCallbackT", bound=TrackingCallback[Any, Dict[str, ndarray]]
)
plotCollectionTrackingCallbackT = TypeVar(
    "plotCollectionTrackingCallbackT", bound=TrackingCallback[Any, Dict[str, Figure]]
)


class TrackingCallbackHandler(
    CacheCallbackHandler[TrackingCallbackT, CallbackResourcesT, CacheDataT],
    Generic[TrackingCallbackT, CallbackResourcesT, CacheDataT],
):
    def __init__(
        self,
        ls_callbacks: Optional[List[TrackingCallbackT]] = None,
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
    TrackingCallbackHandler[scoreTrackingCallbackT, CallbackResourcesT, float],
    Generic[scoreTrackingCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(cache: Dict[str, float], tracking_client: TrackingClient):
        for score_name, score_value in cache.items():
            tracking_client.log_score(score=score_value, name=score_name)


class ArrayCallbackHandler(
    TrackingCallbackHandler[arrayTrackingCallbackT, CallbackResourcesT, ndarray],
    Generic[arrayTrackingCallbackT, CallbackResourcesT],
):
    @staticmethod
    def _export(cache: Dict[str, ndarray], tracking_client: TrackingClient):
        for array_name, array in cache.items():
            tracking_client.log_array(array=array, name=array_name)


class PlotCallbackHandler(
    TrackingCallbackHandler[plotTrackingCallbackT, CallbackResourcesT, Figure],
    Generic[plotTrackingCallbackT, CallbackResourcesT],
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
        scoreCollectionTrackingCallbackT,
        CallbackResourcesT,
        Dict[str, float],
    ],
    Generic[scoreCollectionTrackingCallbackT, CallbackResourcesT],
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
        arrayCollectionTrackingCallbackT,
        CallbackResourcesT,
        Dict[str, ndarray],
    ],
    Generic[arrayCollectionTrackingCallbackT, CallbackResourcesT],
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
        plotCollectionTrackingCallbackT,
        CallbackResourcesT,
        Dict[str, Figure],
    ],
    Generic[plotCollectionTrackingCallbackT, CallbackResourcesT],
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
