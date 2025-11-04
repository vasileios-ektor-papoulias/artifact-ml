from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Dict, Generic, Optional, TypeVar

from artifact_experiment.base.callbacks.cache import CacheCallback, CacheCallbackResources
from artifact_experiment.base.entities.data_split import DataSplit, DataSplitSuffixAppender
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray


@dataclass(frozen=True)
class TrackingCallbackResources(CacheCallbackResources):
    _: KW_ONLY
    data_split: Optional[DataSplit] = None


TrackingCallbackResourcesTContr = TypeVar(
    "TrackingCallbackResourcesTContr", bound=TrackingCallbackResources, contravariant=True
)
CacheDataT = TypeVar("CacheDataT")


class TrackingCallback(
    CacheCallback[TrackingCallbackResourcesTContr, CacheDataT],
    Generic[TrackingCallbackResourcesTContr, CacheDataT],
):
    def __init__(
        self,
        base_key: str,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(base_key=base_key)
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
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def execute(self, resources: TrackingCallbackResourcesTContr):
        super().execute(resources=resources)
        if self._tracking_client is not None:
            assert self.value is not None
            self._export(key=self.key, value=self.value, tracking_client=self._tracking_client)

    @classmethod
    def _qualify_base_key(cls, base_key: str, resources: TrackingCallbackResourcesTContr) -> str:
        key = super()._qualify_base_key(base_key=base_key, resources=resources)
        if resources.data_split is not None:
            key = DataSplitSuffixAppender.append_suffix(name=key, data_split=resources.data_split)
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
    TrackingCallback[TrackingCallbackResourcesTContr, float],
    Generic[TrackingCallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> float: ...


class ArrayCallback(
    ArrayExportMixin,
    TrackingCallback[TrackingCallbackResourcesTContr, ndarray],
    Generic[TrackingCallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> ndarray: ...


class PlotCallback(
    PlotExportMixin,
    TrackingCallback[TrackingCallbackResourcesTContr, Figure],
    Generic[TrackingCallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> Figure: ...


class ScoreCollectionCallback(
    ScoreCollectionExportMixin,
    TrackingCallback[TrackingCallbackResourcesTContr, Dict[str, float]],
    Generic[TrackingCallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> Dict[str, float]: ...


class ArrayCollectionCallback(
    ArrayCollectionExportMixin,
    TrackingCallback[TrackingCallbackResourcesTContr, Dict[str, ndarray]],
    Generic[TrackingCallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> Dict[str, ndarray]: ...


class PlotCollectionCallback(
    PlotCollectionExportMixin,
    TrackingCallback[TrackingCallbackResourcesTContr, Dict[str, Figure]],
    Generic[TrackingCallbackResourcesTContr],
):
    @abstractmethod
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> Dict[str, Figure]: ...
