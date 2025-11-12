from dataclasses import dataclass
from typing import Dict, Generic, TypeVar

from artifact_core._base.primitives import ArtifactResult

from artifact_experiment.base.entities.file import File
from artifact_experiment.base.entities.tracking_data import TrackingData

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
class ScoreQueueItem(ArtifactQueueItem[float]):
    pass


@dataclass(frozen=True)
class ArrayQueueItem(ArtifactQueueItem[Array]):
    pass


@dataclass(frozen=True)
class PlotQueueItem(ArtifactQueueItem[Figure]):
    pass


@dataclass(frozen=True)
class ScoreCollectionQueueItem(ArtifactQueueItem[Dict[str, float]]):
    pass


@dataclass(frozen=True)
class ArrayCollectionQueueItem(ArtifactQueueItem[Dict[str, Array]]):
    pass


@dataclass(frozen=True)
class PlotCollectionQueueItem(ArtifactQueueItem[Dict[str, Figure]]):
    pass


@dataclass(frozen=True)
class FileQueueItem(TrackingQueueItem[File]):
    pass


@dataclass(frozen=True)
class StopFlag(TrackingQueueItem[None]):
    name: str = "stop_flag"
    value: None = None
