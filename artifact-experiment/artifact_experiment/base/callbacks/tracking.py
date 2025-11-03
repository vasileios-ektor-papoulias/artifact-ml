from abc import abstractmethod
from typing import Dict, Generic, Optional, TypeVar

from artifact_experiment.base.callbacks.base import (
    CallbackResources,
)
from artifact_experiment.base.callbacks.cache import (
    CacheCallback,
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
