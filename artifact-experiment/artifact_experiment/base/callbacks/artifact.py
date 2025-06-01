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

resourceSpecProtocolT = TypeVar("resourceSpecProtocolT", bound=ResourceSpecProtocol)
artifactResourcesT = TypeVar("artifactResourcesT", bound=ArtifactResources)
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


@dataclass
class ArtifactCallbackResources(CallbackResources, Generic[artifactResourcesT]):
    artifact_resources: artifactResourcesT


class ArtifactCallback(
    TrackingCallback[ArtifactCallbackResources[artifactResourcesT], artifactResultT],
    Generic[artifactResourcesT, artifactResultT, resourceSpecProtocolT],
):
    def __init__(
        self,
        key: str,
        artifact: Artifact[artifactResourcesT, artifactResultT, Any, resourceSpecProtocolT],
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._artifact = artifact

    @staticmethod
    @abstractmethod
    def _export(key: str, value: artifactResultT, tracking_client: TrackingClient): ...

    def _compute(self, resources: ArtifactCallbackResources[artifactResourcesT]) -> artifactResultT:
        result = self._artifact.compute(resources=resources.artifact_resources)
        return result


class ArtifactScoreCallback(
    ScoreExportMixin,
    ArtifactCallback[artifactResourcesT, float, resourceSpecProtocolT],
):
    pass


class ArtifactArrayCallback(
    ArrayExportMixin,
    ArtifactCallback[artifactResourcesT, ndarray, resourceSpecProtocolT],
):
    pass


class ArtifactPlotCallback(
    PlotExportMixin,
    ArtifactCallback[artifactResourcesT, Figure, resourceSpecProtocolT],
):
    pass


class ArtifactScoreCollectionCallback(
    ScoreCollectionExportMixin,
    ArtifactCallback[artifactResourcesT, Dict[str, float], resourceSpecProtocolT],
):
    pass


class ArtifactArrayCollectionCallback(
    ArrayCollectionExportMixin,
    ArtifactCallback[artifactResourcesT, Dict[str, ndarray], resourceSpecProtocolT],
):
    pass


class ArtifactPlotCollectionCallback(
    PlotCollectionExportMixin,
    ArtifactCallback[artifactResourcesT, Dict[str, Figure], resourceSpecProtocolT],
):
    pass
