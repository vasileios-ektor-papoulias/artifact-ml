from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.components.callbacks.cache import CacheCallback
from artifact_experiment._base.components.resources.tracking import TrackingCallbackResources
from artifact_experiment._base.tracking.background.writer import TrackingQueueWriter
from artifact_experiment._base.typing.tracking_data import TrackingData

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
            key = resources.data_split.append_to(name=key)
        return key


TrackingScoreCallback = TrackingCallback[TrackingCallbackResourcesTContr, Score]
TrackingArrayCallback = TrackingCallback[TrackingCallbackResourcesTContr, Array]
TrackingPlotCallback = TrackingCallback[TrackingCallbackResourcesTContr, Plot]
TrackingScoreCollectionCallback = TrackingCallback[TrackingCallbackResourcesTContr, ScoreCollection]
TrackingArrayCollectionCallback = TrackingCallback[TrackingCallbackResourcesTContr, ArrayCollection]
TrackingPlotCollectionCallback = TrackingCallback[TrackingCallbackResourcesTContr, PlotCollection]
