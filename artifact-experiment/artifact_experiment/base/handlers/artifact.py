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

ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactCallbackHandler(
    TrackingCallbackHandler[
        ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArtifactResultT],
        ArtifactCallbackResources[ArtifactResourcesTContr],
        Any,
    ],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr, ArtifactResultT],
):
    pass


class ArtifactScoreHandler(
    ScoreHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr, float],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    pass


class ArtifactArrayHandler(
    ArrayHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr, ndarray],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    pass


class ArtifactPlotHandler(
    PlotHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Figure],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    pass


class ArtifactScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, float]],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    pass


class ArtifactArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    ArtifactCallbackHandler[
        ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, ndarray]
    ],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    pass


class ArtifactPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    ArtifactCallbackHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Dict[str, Figure]],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    pass


ArtifactCallbackHandlerSuiteT = TypeVar(
    "ArtifactCallbackHandlerSuiteT", bound="ArtifactCallbackHandlerSuite"
)


@dataclass(frozen=True)
class ArtifactCallbackHandlerSuite(
    CallbackHandlerSuite[
        ArtifactCallbackHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any]
    ],
    Generic[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
):
    score_handler: ArtifactScoreHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    array_handler: ArtifactArrayHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    plot_handler: ArtifactPlotHandler[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    score_collection_handler: ArtifactScoreCollectionHandler[
        ArtifactResourcesTContr, ResourceSpecProtocolTContr
    ]
    array_collection_handler: ArtifactArrayCollectionHandler[
        ArtifactResourcesTContr, ResourceSpecProtocolTContr
    ]
    plot_collection_handler: ArtifactPlotCollectionHandler[
        ArtifactResourcesTContr, ResourceSpecProtocolTContr
    ]

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
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactCallbackHandlerSuiteT:
        handler_suite = cls(
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
        return handler_suite
