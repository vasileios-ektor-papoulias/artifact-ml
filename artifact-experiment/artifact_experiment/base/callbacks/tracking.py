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

CallbackResourcesTContr = TypeVar(
    "CallbackResourcesTContr", bound=CallbackResources, contravariant=True
)
CacheDataT = TypeVar("CacheDataT")


class TrackingCallback(
    CacheCallback[CallbackResourcesTContr, CacheDataT],
    Generic[CallbackResourcesTContr, CacheDataT],
):
    def __init__(self, key: str, tracking_client: Optional[TrackingClient] = None):
        super().__init__(key=key)
        self._tracking_client = tracking_client

    @property
    def tracking_client(self) -> Optional[TrackingClient]:
        return self._tracking_client

    @tracking_client.setter
    def tracking_client(self, tracking_client: Optional[TrackingClient]):
        self._tracking_client = tracking_client

    @property
    def tracking_enabled(self) -> bool:
        return self._tracking_client is not None

    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def execute(self, resources: CallbackResourcesTContr):
        super().execute(resources=resources)
        if self._tracking_client is not None:
            assert self.value is not None
            self._export(key=self.key, value=self.value, tracking_client=self._tracking_client)


class ScoreExportMixin:
    @staticmethod
    def _export(key: str, value: float, tracking_client: TrackingClient):
        tracking_client.log_score(score=value, name=key)


class ArrayExportMixin:
    @staticmethod
    def _export(key: str, value: ndarray, tracking_client: TrackingClient):
        tracking_client.log_array(array=value, name=key)


class PlotExportMixin:
    @staticmethod
    def _export(key: str, value: Figure, tracking_client: TrackingClient):
        tracking_client.log_plot(plot=value, name=key)


class ScoreCollectionExportMixin:
    @staticmethod
    def _export(key: str, value: Dict[str, float], tracking_client: TrackingClient):
        tracking_client.log_score_collection(score_collection=value, name=key)


class ArrayCollectionExportMixin:
    @staticmethod
    def _export(key: str, value: Dict[str, ndarray], tracking_client: TrackingClient):
        tracking_client.log_array_collection(array_collection=value, name=key)


class PlotCollectionExportMixin:
    @staticmethod
    def _export(key: str, value: Dict[str, Figure], tracking_client: TrackingClient):
        tracking_client.log_plot_collection(plot_collection=value, name=key)


class ScoreCallback(
    ScoreExportMixin,
    TrackingCallback[CallbackResourcesTContr, float],
    Generic[CallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> float: ...


class ArrayCallback(
    ArrayExportMixin,
    TrackingCallback[CallbackResourcesTContr, ndarray],
    Generic[CallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> ndarray: ...


class PlotCallback(
    PlotExportMixin,
    TrackingCallback[CallbackResourcesTContr, Figure],
    Generic[CallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> Figure: ...


class ScoreCollectionCallback(
    ScoreCollectionExportMixin,
    TrackingCallback[CallbackResourcesTContr, Dict[str, float]],
    Generic[CallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> Dict[str, float]: ...


class ArrayCollectionCallback(
    ArrayCollectionExportMixin,
    TrackingCallback[CallbackResourcesTContr, Dict[str, ndarray]],
    Generic[CallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> Dict[str, ndarray]: ...


class PlotCollectionCallback(
    PlotCollectionExportMixin,
    TrackingCallback[CallbackResourcesTContr, Dict[str, Figure]],
    Generic[CallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: CallbackResourcesTContr) -> Dict[str, Figure]: ...


TrackingCallbackT = TypeVar("TrackingCallbackT", bound=TrackingCallback)
ScoreCallbackT = TypeVar("ScoreCallbackT", bound=TrackingCallback[Any, float])
ArrayCallbackT = TypeVar("ArrayCallbackT", bound=TrackingCallback[Any, ndarray])
PlotCallbackT = TypeVar("PlotCallbackT", bound=TrackingCallback[Any, Figure])
ScoreCollectionCallbackT = TypeVar(
    "ScoreCollectionCallbackT", bound=TrackingCallback[Any, Dict[str, float]]
)
ArrayCollectionCallbackT = TypeVar(
    "ArrayCollectionCallbackT", bound=TrackingCallback[Any, Dict[str, ndarray]]
)
PlotCollectionCallbackT = TypeVar(
    "PlotCollectionCallbackT", bound=TrackingCallback[Any, Dict[str, Figure]]
)


class TrackingCallbackHandler(
    CacheCallbackHandler[TrackingCallbackT, CallbackResourcesTContr, CacheDataT],
    Generic[TrackingCallbackT, CallbackResourcesTContr, CacheDataT],
):
    def __init__(
        self,
        ls_callbacks: Optional[List[TrackingCallbackT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(ls_callbacks=ls_callbacks)
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
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient):
        pass

    def execute(self, resources: CallbackResourcesTContr):
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
    def _invalidate_callback_tracking_clients(ls_callbacks: List[TrackingCallbackT]):
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


class ScoreCallbackHandler(
    ScoreHandlerExportMixin,
    TrackingCallbackHandler[ScoreCallbackT, CallbackResourcesTContr, float],
    Generic[ScoreCallbackT, CallbackResourcesTContr],
):
    pass


class ArrayCallbackHandler(
    ArrayHandlerExportMixin,
    TrackingCallbackHandler[ArrayCallbackT, CallbackResourcesTContr, ndarray],
    Generic[ArrayCallbackT, CallbackResourcesTContr],
):
    pass


class PlotCallbackHandler(
    PlotHandlerExportMixin,
    TrackingCallbackHandler[PlotCallbackT, CallbackResourcesTContr, Figure],
    Generic[PlotCallbackT, CallbackResourcesTContr],
):
    pass


class ScoreCollectionCallbackHandler(
    ScoreCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        ScoreCollectionCallbackT,
        CallbackResourcesTContr,
        Dict[str, float],
    ],
    Generic[ScoreCollectionCallbackT, CallbackResourcesTContr],
):
    pass


class ArrayCollectionCallbackHandler(
    ArrayCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        ArrayCollectionCallbackT,
        CallbackResourcesTContr,
        Dict[str, ndarray],
    ],
    Generic[ArrayCollectionCallbackT, CallbackResourcesTContr],
):
    pass


class PlotCollectionCallbackHandler(
    PlotCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        PlotCollectionCallbackT,
        CallbackResourcesTContr,
        Dict[str, Figure],
    ],
    Generic[PlotCollectionCallbackT, CallbackResourcesTContr],
):
    pass
