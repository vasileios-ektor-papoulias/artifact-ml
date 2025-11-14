from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.typing.tracking_data import TrackingData

TrackingDataTCov = TypeVar("TrackingDataTCov", bound=TrackingData, covariant=True)


@dataclass(frozen=True)
class TrackingQueueItem(Generic[TrackingDataTCov]):
    name: str
    value: TrackingDataTCov


ArtifactResultTCov = TypeVar("ArtifactResultTCov", bound=ArtifactResult, covariant=True)


@dataclass(frozen=True)
class ArtifactQueueItem(TrackingQueueItem[ArtifactResultTCov]):
    pass


@dataclass(frozen=True)
class ScoreQueueItem(ArtifactQueueItem[Score]):
    pass


@dataclass(frozen=True)
class ArrayQueueItem(ArtifactQueueItem[Array]):
    pass


@dataclass(frozen=True)
class PlotQueueItem(ArtifactQueueItem[Plot]):
    pass


@dataclass(frozen=True)
class ScoreCollectionQueueItem(ArtifactQueueItem[ScoreCollection]):
    pass


@dataclass(frozen=True)
class ArrayCollectionQueueItem(ArtifactQueueItem[ArrayCollection]):
    pass


@dataclass(frozen=True)
class PlotCollectionQueueItem(ArtifactQueueItem[PlotCollection]):
    pass


@dataclass(frozen=True)
class FileQueueItem(TrackingQueueItem[File]):
    pass


@dataclass(frozen=True)
class StopFlag(TrackingQueueItem[None]):
    name: str = "stop_flag"
    value: None = None
