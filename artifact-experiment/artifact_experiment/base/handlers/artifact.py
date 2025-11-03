from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactCallback,
    ArtifactCallbackResources,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.base.handlers.base import CallbackHandlerSuite
from artifact_experiment.base.handlers.tracking import (
    ArrayCollectionHandlerExportMixin,
    ArrayHandlerExportMixin,
    PlotCollectionHandlerExportMixin,
    PlotHandlerExportMixin,
    ScoreCollectionHandlerExportMixin,
    ScoreHandlerExportMixin,
    TrackingCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactCallbackHandler(
    TrackingCallbackHandler[
        ArtifactCallback[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactResultT],
        ArtifactCallbackResources[ArtifactResourcesT],
        Any,
    ],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactResultT],
):
    pass


class ArtifactScoreHandler(
    ScoreHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesT, ResourceSpecProtocolT, float],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT],
):
    pass


class ArtifactArrayHandler(
    ArrayHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesT, ResourceSpecProtocolT, ndarray],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT],
):
    pass


class ArtifactPlotHandler(
    PlotHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesT, ResourceSpecProtocolT, Figure],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT],
):
    pass


class ArtifactScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesT, ResourceSpecProtocolT, Dict[str, float]],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT],
):
    pass


class ArtifactArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesT, ResourceSpecProtocolT, Dict[str, ndarray]],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT],
):
    pass


class ArtifactPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesT, ResourceSpecProtocolT, Dict[str, Figure]],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT],
):
    pass


ArtifactHandlerSuiteT = TypeVar("ArtifactHandlerSuiteT", bound="ArtifactHandlerSuite")


@dataclass(frozen=True)
class ArtifactHandlerSuite(
    CallbackHandlerSuite[ArtifactCallbackHandler[ArtifactResourcesT, ResourceSpecProtocolT, Any]],
    Generic[ArtifactResourcesT, ResourceSpecProtocolT],
):
    score_handler: ArtifactScoreHandler[ArtifactResourcesT, ResourceSpecProtocolT]
    array_handler: ArtifactArrayHandler[ArtifactResourcesT, ResourceSpecProtocolT]
    plot_handler: ArtifactPlotHandler[ArtifactResourcesT, ResourceSpecProtocolT]
    score_collection_handler: ArtifactScoreCollectionHandler[
        ArtifactResourcesT, ResourceSpecProtocolT
    ]
    array_collection_handler: ArtifactArrayCollectionHandler[
        ArtifactResourcesT, ResourceSpecProtocolT
    ]
    plot_collection_handler: ArtifactPlotCollectionHandler[
        ArtifactResourcesT, ResourceSpecProtocolT
    ]

    @classmethod
    def build(
        cls: Type[ArtifactHandlerSuiteT],
        score_callbacks: Sequence[ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT]],
        array_callbacks: Sequence[ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT]],
        plot_callbacks: Sequence[ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT]],
        score_collection_callbacks: Sequence[
            ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]
        ],
        array_collection_callbacks: Sequence[
            ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]
        ],
        plot_collection_callbacks: Sequence[
            ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]
        ],
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactHandlerSuiteT:
        handler_collection = cls(
            score_handler=ArtifactScoreHandler(
                callbacks=score_callbacks, tracking_client=tracking_client
            ),
            array_handler=ArtifactArrayHandler(
                callbacks=array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=ArtifactPlotHandler(
                callbacks=plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=ArtifactScoreCollectionHandler(
                callbacks=score_collection_callbacks, tracking_client=tracking_client
            ),
            array_collection_handler=ArtifactArrayCollectionHandler(
                callbacks=array_collection_callbacks, tracking_client=tracking_client
            ),
            plot_collection_handler=ArtifactPlotCollectionHandler(
                callbacks=plot_collection_callbacks, tracking_client=tracking_client
            ),
        )
        return handler_collection
