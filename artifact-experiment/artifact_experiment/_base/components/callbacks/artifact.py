from typing import Any, Generic, Optional, TypeVar

from artifact_core.spi.artifact import Artifact
from artifact_core.spi.resources import ArtifactResources, ResourceSpecProtocol
from artifact_core.typing import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.components.callbacks.tracking import TrackingCallback
from artifact_experiment._base.components.resources.artifact import ArtifactCallbackResources
from artifact_experiment._base.tracking.background.writer import TrackingQueueWriter

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


ArtifactScoreCallback = ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Score]
ArtifactArrayCallback = ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Array]
ArtifactPlotCallback = ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Plot]
ArtifactScoreCollectionCallback = ArtifactCallback[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, ScoreCollection
]
ArtifactArrayCollectionCallback = ArtifactCallback[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArrayCollection
]
ArtifactPlotCollectionCallback = ArtifactCallback[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, PlotCollection
]
