from typing import Any, TypeVar

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

from artifact_experiment._base.components.callbacks.artifact import (
    ArtifactCallback,
    ArtifactCallbackResources,
)
from artifact_experiment._base.components.handlers.tracking import TrackingCallbackHandler

ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ArtifactResultTCov = TypeVar("ArtifactResultTCov", bound=ArtifactResult, covariant=True)


ArtifactCallbackHandler = TrackingCallbackHandler[
    ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArtifactResultTCov],
    ArtifactCallbackResources[ArtifactResourcesTContr],
    Any,
]


ArtifactScoreHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Score
]
ArtifactArrayHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Array
]
ArtifactPlotHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, Plot
]
ArtifactScoreCollectionHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, ScoreCollection
]
ArtifactArrayCollectionHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArrayCollection
]
ArtifactPlotCollectionHandler = ArtifactCallbackHandler[
    ArtifactResourcesTContr, ResourceSpecProtocolTContr, PlotCollection
]
