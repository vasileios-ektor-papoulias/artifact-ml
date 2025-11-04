from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.callbacks.tracking import TrackingCallback, TrackingCallbackResources
from artifact_experiment.base.handlers.cache import CacheCallbackHandler
from artifact_experiment.base.tracking.client import TrackingClient

TrackingCallbackTCov = TypeVar(
    "TrackingCallbackTCov", bound=TrackingCallback[Any, Any], covariant=True
)
TrackingCallbackResourcesTContr = TypeVar(
    "TrackingCallbackResourcesTContr", bound=TrackingCallbackResources, contravariant=True
)
CacheDataT = TypeVar("CacheDataT", covariant=True)


class TrackingCallbackHandler(
    CacheCallbackHandler[TrackingCallbackTCov, TrackingCallbackResourcesTContr, CacheDataT],
    Generic[TrackingCallbackTCov, TrackingCallbackResourcesTContr, CacheDataT],
):
    def __init__(
        self,
        callbacks: Optional[Sequence[TrackingCallbackTCov]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(callbacks=callbacks)
        self._tracking_client = tracking_client

    @property
    def tracking_client(self) -> Optional[TrackingClient]:
        return self._tracking_client

    @tracking_client.setter
    def tracking_client(self, tracking_client: Optional[TrackingClient]):
        self._invalidate_callback_tracking_clients(ls_callbacks=self._ls_callbacks)
        self._tracking_client = tracking_client

    @property
    def tracking_enabled(self) -> bool:
        return self._tracking_client is not None

    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...

    def execute(self, resources: TrackingCallbackResourcesTContr):
        if self._tracking_client is not None:
            self._invalidate_callback_tracking_clients(ls_callbacks=self._ls_callbacks)
            super().execute(resources=resources)
            self._export(
                cache=self.active_cache,
                tracking_client=self._tracking_client,
            )
        else:
            super().execute(resources=resources)

    @staticmethod
    def _invalidate_callback_tracking_clients(ls_callbacks: List[TrackingCallbackTCov]):
        for callback in ls_callbacks:
            callback.tracking_client = None


class ScoreHandlerExportMixin:
    @staticmethod
    def _export(cache: Dict[str, float], tracking_client: TrackingClient):
        for score_name, score_value in cache.items():
            tracking_client.log_score(score=score_value, name=score_name)


class ArrayHandlerExportMixin:
    @staticmethod
    def _export(cache: Dict[str, ndarray], tracking_client: TrackingClient):
        for array_name, array in cache.items():
            tracking_client.log_array(array=array, name=array_name)


class PlotHandlerExportMixin:
    @staticmethod
    def _export(
        cache: Dict[str, Figure],
        tracking_client: TrackingClient,
    ):
        for plot_name, plot in cache.items():
            tracking_client.log_plot(plot=plot, name=plot_name)


class ScoreCollectionHandlerExportMixin:
    @staticmethod
    def _export(
        cache: Dict[str, Dict[str, float]],
        tracking_client: TrackingClient,
    ):
        for score_collection_name, score_collection in cache.items():
            tracking_client.log_score_collection(
                score_collection=score_collection, name=score_collection_name
            )


class ArrayCollectionHandlerExportMixin:
    @staticmethod
    def _export(
        cache: Dict[str, Dict[str, ndarray]],
        tracking_client: TrackingClient,
    ):
        for array_collection_name, array_collection in cache.items():
            tracking_client.log_array_collection(
                array_collection=array_collection, name=array_collection_name
            )


class PlotCollectionHandlerExportMixin:
    @staticmethod
    def _export(
        cache: Dict[str, Dict[str, Figure]],
        tracking_client: TrackingClient,
    ):
        for plot_collection_name, plot_collection in cache.items():
            tracking_client.log_plot_collection(
                plot_collection=plot_collection, name=plot_collection_name
            )


ScoreCallbackTCov = TypeVar("ScoreCallbackTCov", bound=TrackingCallback[Any, float], covariant=True)


class ScoreCallbackHandler(
    ScoreHandlerExportMixin,
    TrackingCallbackHandler[ScoreCallbackTCov, TrackingCallbackResourcesTContr, float],
    Generic[ScoreCallbackTCov, TrackingCallbackResourcesTContr],
):
    pass


ArrayCallbackTCov = TypeVar(
    "ArrayCallbackTCov", bound=TrackingCallback[Any, ndarray], covariant=True
)


class ArrayCallbackHandler(
    ArrayHandlerExportMixin,
    TrackingCallbackHandler[ArrayCallbackTCov, TrackingCallbackResourcesTContr, ndarray],
    Generic[ArrayCallbackTCov, TrackingCallbackResourcesTContr],
):
    pass


PlotCallbackTCov = TypeVar("PlotCallbackTCov", bound=TrackingCallback[Any, Figure], covariant=True)


class PlotCallbackHandler(
    PlotHandlerExportMixin,
    TrackingCallbackHandler[PlotCallbackTCov, TrackingCallbackResourcesTContr, Figure],
    Generic[PlotCallbackTCov, TrackingCallbackResourcesTContr],
):
    pass


ScoreCollectionCallbackTCov = TypeVar(
    "ScoreCollectionCallbackTCov", bound=TrackingCallback[Any, Dict[str, float]], covariant=True
)


class ScoreCollectionCallbackHandler(
    ScoreCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        ScoreCollectionCallbackTCov,
        TrackingCallbackResourcesTContr,
        Dict[str, float],
    ],
    Generic[ScoreCollectionCallbackTCov, TrackingCallbackResourcesTContr],
):
    pass


ArrayCollectionCallbackTCov = TypeVar(
    "ArrayCollectionCallbackTCov", bound=TrackingCallback[Any, Dict[str, ndarray]], covariant=True
)


class ArrayCollectionCallbackHandler(
    ArrayCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        ArrayCollectionCallbackTCov,
        TrackingCallbackResourcesTContr,
        Dict[str, ndarray],
    ],
    Generic[ArrayCollectionCallbackTCov, TrackingCallbackResourcesTContr],
):
    pass


PlotCollectionCallbackTCov = TypeVar(
    "PlotCollectionCallbackTCov", bound=TrackingCallback[Any, Dict[str, Figure]], covariant=True
)


class PlotCollectionCallbackHandler(
    PlotCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        PlotCollectionCallbackTCov,
        TrackingCallbackResourcesTContr,
        Dict[str, Figure],
    ],
    Generic[PlotCollectionCallbackTCov, TrackingCallbackResourcesTContr],
):
    pass
