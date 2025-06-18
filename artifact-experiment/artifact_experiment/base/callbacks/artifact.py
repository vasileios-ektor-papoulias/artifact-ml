from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_experiment.base.callbacks.base import (
    CallbackResources,
)
from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayExportMixin,
    PlotCollectionExportMixin,
    PlotExportMixin,
    ScoreCollectionExportMixin,
    ScoreExportMixin,
    TrackingCallback,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


@dataclass
class ArtifactCallbackResources(CallbackResources, Generic[ArtifactResourcesT]):
    artifact_resources: ArtifactResourcesT


class ArtifactCallback(
    TrackingCallback[ArtifactCallbackResources[ArtifactResourcesT], ArtifactResultT],
    Generic[ArtifactResourcesT, ArtifactResultT, ResourceSpecProtocolT],
):
    def __init__(
        self,
        key: str,
        artifact: Artifact[ArtifactResourcesT, ArtifactResultT, Any, ResourceSpecProtocolT],
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._artifact = artifact

    @staticmethod
    @abstractmethod
    def _export(key: str, value: ArtifactResultT, tracking_client: TrackingClient): ...

    def _compute(self, resources: ArtifactCallbackResources[ArtifactResourcesT]) -> ArtifactResultT:
        result = self._artifact.compute(resources=resources.artifact_resources)
        return result


class ArtifactScoreCallback(
    ScoreExportMixin,
    ArtifactCallback[ArtifactResourcesT, float, ResourceSpecProtocolT],
):
    pass


class ArtifactArrayCallback(
    ArrayExportMixin,
    ArtifactCallback[ArtifactResourcesT, ndarray, ResourceSpecProtocolT],
):
    pass


class ArtifactPlotCallback(
    PlotExportMixin,
    ArtifactCallback[ArtifactResourcesT, Figure, ResourceSpecProtocolT],
):
    pass


class ArtifactScoreCollectionCallback(
    ScoreCollectionExportMixin,
    ArtifactCallback[ArtifactResourcesT, Dict[str, float], ResourceSpecProtocolT],
):
    pass


class ArtifactArrayCollectionCallback(
    ArrayCollectionExportMixin,
    ArtifactCallback[ArtifactResourcesT, Dict[str, ndarray], ResourceSpecProtocolT],
):
    pass


class ArtifactPlotCollectionCallback(
    PlotCollectionExportMixin,
    ArtifactCallback[ArtifactResourcesT, Dict[str, Figure], ResourceSpecProtocolT],
):
    pass
