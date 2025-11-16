from typing import Any, Generic, Sequence, Type, TypeVar

from artifact_core.spi.resources import ArtifactResources, ResourceSpecProtocol

from artifact_experiment._base.components.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactCallback,
    ArtifactCallbackResources,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment._base.components.handler_suites.tracking import (
    TrackingCallbackHandlerSuite,
)
from artifact_experiment._base.components.handlers.artifact import ArtifactCallbackHandler

ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ArtifactCallbackHandlerSuiteT = TypeVar(
    "ArtifactCallbackHandlerSuiteT", bound="ArtifactCallbackHandlerSuite"
)


class ArtifactCallbackHandlerSuite(
    TrackingCallbackHandlerSuite[
        ArtifactCallbackHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any],
        ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any],
        ArtifactCallbackResources[ArtifactResourcesTContr],
    ],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    @classmethod
    def build(
        cls: Type[ArtifactCallbackHandlerSuiteT],
        score_callbacks: Sequence[
            ArtifactScoreCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
        ],
        array_callbacks: Sequence[
            ArtifactArrayCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
        ],
        plot_callbacks: Sequence[
            ArtifactPlotCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
        ],
        score_collection_callbacks: Sequence[
            ArtifactScoreCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
        ],
        array_collection_callbacks: Sequence[
            ArtifactArrayCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
        ],
        plot_collection_callbacks: Sequence[
            ArtifactPlotCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
        ],
    ) -> ArtifactCallbackHandlerSuiteT:
        handler_suite = cls(
            score_handler=ArtifactCallbackHandler(callbacks=score_callbacks),
            array_handler=ArtifactCallbackHandler(callbacks=array_callbacks),
            plot_handler=ArtifactCallbackHandler(callbacks=plot_callbacks),
            score_collection_handler=ArtifactCallbackHandler(callbacks=score_collection_callbacks),
            array_collection_handler=ArtifactCallbackHandler(callbacks=array_collection_callbacks),
            plot_collection_handler=ArtifactCallbackHandler(callbacks=plot_collection_callbacks),
        )
        return handler_suite
