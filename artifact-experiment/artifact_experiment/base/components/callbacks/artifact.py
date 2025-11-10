from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_experiment.base.components.callbacks.tracking import (
    TrackingCallback,
    TrackingCallbackResources,
)
from artifact_experiment.base.tracking.background.writer import TrackingQueueWriter
from matplotlib.figure import Figure
from numpy import ndarray

ArtifactResourcesTCov = TypeVar("ArtifactResourcesTCov", bound=ArtifactResources, covariant=True)


@dataclass(frozen=True)
class ArtifactCallbackResources(TrackingCallbackResources, Generic[ArtifactResourcesTCov]):
    artifact_resources: ArtifactResourcesTCov


ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ArtifactResultTCov = TypeVar("ArtifactResultTCov", bound=ArtifactResult, covariant=True)


class ArtifactCallback(
    TrackingCallback[ArtifactCallbackResources[ArtifactResourcesTContr], ArtifactResultTCov],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArtifactResultTCov],
):
    def __init__(
        self,
        base_key: str,
        artifact: Artifact[
            ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any, ArtifactResultTCov
        ],
        writer: Optional[TrackingQueueWriter[ArtifactResultTCov]] = None,
    ):
        super().__init__(base_key=base_key, writer=writer)
        self._artifact = artifact

    def _compute(
        self, resources: ArtifactCallbackResources[ArtifactResourcesTContr]
    ) -> ArtifactResultTCov:
        result = self._artifact.compute(resources=resources.artifact_resources)
        return result


ArtifactScoreCallback = ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, float]
ArtifactArrayCallback = ArtifactCallback[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, ndarray
]
ArtifactPlotCallback = ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Figure]
ArtifactScoreCollectionCallback = ArtifactCallback[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, float]
]
ArtifactArrayCollectionCallback = ArtifactCallback[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, ndarray]
]
ArtifactPlotCollectionCallback = ArtifactCallback[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, Figure]
]
