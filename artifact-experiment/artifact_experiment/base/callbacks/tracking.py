from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar

from artifact_experiment.base.callbacks.base import (
    CallbackResources,
)
from artifact_experiment.base.callbacks.cache import (
    CacheCallback,
    CacheCallbackHandler,
)
from artifact_experiment.base.entities.data_split import DataSplit, DataSplitSuffixAppender
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
    def __init__(
        self,
        name: str,
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        key = self._get_key(name=name, data_split=data_split)
        super().__init__(key=key)
        self._data_split = data_split
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

    @property
    def data_split(self) -> Optional[DataSplit]:
        return self._data_split

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

    @classmethod
    def _get_key(cls, name: str, data_split: Optional[DataSplit]) -> str:
        if data_split is not None:
            key = DataSplitSuffixAppender.append_suffix(name=name, data_split=data_split)
        else:
            key = name
        return key


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


TrackingCallbackTCov = TypeVar("TrackingCallbackTCov", bound=TrackingCallback, covariant=True)


class TrackingCallbackHandler(
    CacheCallbackHandler[TrackingCallbackTCov, CallbackResourcesTContr, CacheDataT],
    Generic[TrackingCallbackTCov, CallbackResourcesTContr, CacheDataT],
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
    TrackingCallbackHandler[ScoreCallbackTCov, CallbackResourcesTContr, float],
    Generic[ScoreCallbackTCov, CallbackResourcesTContr],
):
    pass


ArrayCallbackTCov = TypeVar(
    "ArrayCallbackTCov", bound=TrackingCallback[Any, ndarray], covariant=True
)


class ArrayCallbackHandler(
    ArrayHandlerExportMixin,
    TrackingCallbackHandler[ArrayCallbackTCov, CallbackResourcesTContr, ndarray],
    Generic[ArrayCallbackTCov, CallbackResourcesTContr],
):
    pass


PlotCallbackTCov = TypeVar("PlotCallbackTCov", bound=TrackingCallback[Any, Figure], covariant=True)


class PlotCallbackHandler(
    PlotHandlerExportMixin,
    TrackingCallbackHandler[PlotCallbackTCov, CallbackResourcesTContr, Figure],
    Generic[PlotCallbackTCov, CallbackResourcesTContr],
):
    pass


ScoreCollectionCallbackTCov = TypeVar(
    "ScoreCollectionCallbackTCov", bound=TrackingCallback[Any, Dict[str, float]], covariant=True
)


class ScoreCollectionCallbackHandler(
    ScoreCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        ScoreCollectionCallbackTCov,
        CallbackResourcesTContr,
        Dict[str, float],
    ],
    Generic[ScoreCollectionCallbackTCov, CallbackResourcesTContr],
):
    pass


ArrayCollectionCallbackTCov = TypeVar(
    "ArrayCollectionCallbackTCov", bound=TrackingCallback[Any, Dict[str, ndarray]], covariant=True
)


class ArrayCollectionCallbackHandler(
    ArrayCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        ArrayCollectionCallbackTCov,
        CallbackResourcesTContr,
        Dict[str, ndarray],
    ],
    Generic[ArrayCollectionCallbackTCov, CallbackResourcesTContr],
):
    pass


PlotCollectionCallbackTCov = TypeVar(
    "PlotCollectionCallbackTCov", bound=TrackingCallback[Any, Dict[str, Figure]], covariant=True
)


class PlotCollectionCallbackHandler(
    PlotCollectionHandlerExportMixin,
    TrackingCallbackHandler[
        PlotCollectionCallbackTCov,
        CallbackResourcesTContr,
        Dict[str, Figure],
    ],
    Generic[PlotCollectionCallbackTCov, CallbackResourcesTContr],
):
    pass
