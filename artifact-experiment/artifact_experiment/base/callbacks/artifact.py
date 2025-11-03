from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayExportMixin,
    PlotCollectionExportMixin,
    PlotExportMixin,
    ScoreCollectionExportMixin,
    ScoreExportMixin,
    TrackingCallback,
)
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

ArtifactResourcesTCov = TypeVar("ArtifactResourcesTCov", bound=ArtifactResources, covariant=True)


@dataclass
class ArtifactCallbackResources(CallbackResources, Generic[ArtifactResourcesTCov]):
    artifact_resources: ArtifactResourcesTCov


ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactCallback(
    TrackingCallback[ArtifactCallbackResources[ArtifactResourcesTContr], ArtifactResultT],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolT, ArtifactResultT],
):
    def __init__(
        self,
        name: str,
        artifact: Artifact[ArtifactResourcesTContr, ResourceSpecProtocolT, Any, ArtifactResultT],
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(name=name, data_split=data_split, tracking_client=tracking_client)
        self._artifact = artifact

    @staticmethod
    @abstractmethod
    def _export(key: str, value: ArtifactResultT, tracking_client: TrackingClient): ...

    def _compute(
        self, resources: ArtifactCallbackResources[ArtifactResourcesTContr]
    ) -> ArtifactResultT:
        result = self._artifact.compute(resources=resources.artifact_resources)
        return result


class ArtifactScoreCallback(
    ScoreExportMixin,
    ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolT, float],
):
    pass


class ArtifactArrayCallback(
    ArrayExportMixin,
    ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolT, ndarray],
):
    pass


class ArtifactPlotCallback(
    PlotExportMixin,
    ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolT, Figure],
):
    pass


class ArtifactScoreCollectionCallback(
    ScoreCollectionExportMixin,
    ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolT, Dict[str, float]],
):
    pass


class ArtifactArrayCollectionCallback(
    ArrayCollectionExportMixin,
    ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolT, Dict[str, ndarray]],
):
    pass


class ArtifactPlotCollectionCallback(
    PlotCollectionExportMixin,
    ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolT, Dict[str, Figure]],
):
    pass
