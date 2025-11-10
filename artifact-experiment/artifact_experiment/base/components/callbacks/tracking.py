from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Dict, Generic, Optional, TypeVar

from artifact_experiment.base.components.callbacks.cache import (
    CacheCallback,
    CacheCallbackResources,
)
from artifact_experiment.base.entities.data_split import DataSplit, DataSplitSuffixAppender
from artifact_experiment.base.entities.tracking_data import TrackingData
from artifact_experiment.base.tracking.background.writer import TrackingQueueWriter
from matplotlib.figure import Figure
from numpy import ndarray


@dataclass(frozen=True)
class TrackingCallbackResources(CacheCallbackResources):
    _: KW_ONLY
    data_split: Optional[DataSplit] = None


TrackingCallbackResourcesTContr = TypeVar(
    "TrackingCallbackResourcesTContr", bound=TrackingCallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", bound=TrackingData, covariant=True)


class TrackingCallback(
    CacheCallback[TrackingCallbackResourcesTContr, CacheDataTCov],
    Generic[TrackingCallbackResourcesTContr, CacheDataTCov],
):
    def __init__(
        self,
        base_key: str,
        writer: Optional[TrackingQueueWriter[CacheDataTCov]] = None,
    ):
        super().__init__(base_key=base_key)
        self._writer = writer

    @property
    def tracking_enabled(self) -> bool:
        return self._writer is not None

    @abstractmethod
    def _compute(self, resources: TrackingCallbackResourcesTContr) -> CacheDataTCov: ...

    def execute(self, resources: TrackingCallbackResourcesTContr):
        super().execute(resources=resources)
        if self._writer is not None:
            assert self.value is not None
            self._writer.write(name=self.key, value=self.value)

    @classmethod
    def _qualify_base_key(cls, base_key: str, resources: TrackingCallbackResourcesTContr) -> str:
        key = super()._qualify_base_key(base_key=base_key, resources=resources)
        if resources.data_split is not None:
            key = DataSplitSuffixAppender.append_suffix(name=key, data_split=resources.data_split)
        return key


TrackingScoreCallback = TrackingCallback[TrackingCallbackResourcesTContr, float]
TrackingArrayCallback = TrackingCallback[TrackingCallbackResourcesTContr, ndarray]
TrackingPlotCallback = TrackingCallback[TrackingCallbackResourcesTContr, Figure]
TrackingScoreCollectionCallback = TrackingCallback[
    TrackingCallbackResourcesTContr, Dict[str, float]
]
TrackingArrayCollectionCallback = TrackingCallback[
    TrackingCallbackResourcesTContr, Dict[str, ndarray]
]
TrackingPlotCollectionCallback = TrackingCallback[
    TrackingCallbackResourcesTContr, Dict[str, Figure]
]
