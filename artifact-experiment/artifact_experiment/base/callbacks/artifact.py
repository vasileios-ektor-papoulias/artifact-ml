from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_experiment.base.callbacks.base import CallbackHandlerSuite, CallbackResources
from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayCollectionHandlerExportMixin,
    ArrayExportMixin,
    ArrayHandlerExportMixin,
    PlotCollectionExportMixin,
    PlotCollectionHandlerExportMixin,
    PlotExportMixin,
    PlotHandlerExportMixin,
    ScoreCollectionExportMixin,
    ScoreCollectionHandlerExportMixin,
    ScoreExportMixin,
    ScoreHandlerExportMixin,
    TrackingCallback,
    TrackingCallbackHandler,
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


ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)


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
