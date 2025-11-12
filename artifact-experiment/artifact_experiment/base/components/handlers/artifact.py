from typing import Any, Dict, Generic, TypeVar

from artifact_core._base.primitives import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)

from artifact_experiment.base.components.callbacks.artifact import (
    ArtifactCallback,
    ArtifactCallbackResources,
)
from artifact_experiment.base.components.handlers.tracking import TrackingCallbackHandler

ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ArtifactResultTCov = TypeVar("ArtifactResultTCov", bound=ArtifactResult, covariant=True)


class ArtifactCallbackHandler(
    TrackingCallbackHandler[
        ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArtifactResultTCov],
        ArtifactCallbackResources[ArtifactResourcesTContr],
        Any,
    ],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArtifactResultTCov],
):
    pass


ArtifactScoreHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, float
]
ArtifactArrayHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Array
]
ArtifactPlotHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Figure
]
ArtifactScoreCollectionHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, float]
]
ArtifactArrayCollectionHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, Array]
]
ArtifactPlotCollectionHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, Figure]
]
