from abc import abstractmethod
from typing import Dict, Generic, Type, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from matplotlib.figure import Figure
from numpy import ndarray

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactCallbackFactory(
    Generic[
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
        ArtifactResourcesT,
        ResourceSpecProtocolT,
    ]
):
    @staticmethod
    @abstractmethod
    def _get_score_registry() -> Type[
        ArtifactRegistry[ScoreTypeT, ArtifactResourcesT, float, ResourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_registry() -> Type[
        ArtifactRegistry[ArrayTypeT, ArtifactResourcesT, ndarray, ResourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_registry() -> Type[
        ArtifactRegistry[PlotTypeT, ArtifactResourcesT, Figure, ResourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_registry() -> Type[
        ArtifactRegistry[
            ScoreCollectionTypeT, ArtifactResourcesT, Dict[str, float], ResourceSpecProtocolT
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_registry() -> Type[
        ArtifactRegistry[
            ArrayCollectionTypeT, ArtifactResourcesT, Dict[str, ndarray], ResourceSpecProtocolT
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_registry() -> Type[
        ArtifactRegistry[
            PlotCollectionTypeT, ArtifactResourcesT, Dict[str, Figure], ResourceSpecProtocolT
        ]
    ]: ...

    @classmethod
    def build_score_callback(
        cls, score_type: ScoreTypeT, resource_spec: ResourceSpecProtocolT
    ) -> ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_registry()
        artifact = registry.get(artifact_type=score_type, resource_spec=resource_spec)
        callback = ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=score_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_array_callback(
        cls, array_type: ArrayTypeT, resource_spec: ResourceSpecProtocolT
    ) -> ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_registry()
        artifact = registry.get(artifact_type=array_type, resource_spec=resource_spec)
        callback = ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=array_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_plot_callback(
        cls, plot_type: PlotTypeT, resource_spec: ResourceSpecProtocolT
    ) -> ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_registry()
        artifact = registry.get(artifact_type=plot_type, resource_spec=resource_spec)
        callback = ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=plot_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_score_collection_callback(
        cls, score_collection_type: ScoreCollectionTypeT, resource_spec: ResourceSpecProtocolT
    ) -> ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_collection_registry()
        artifact = registry.get(artifact_type=score_collection_type, resource_spec=resource_spec)
        callback = ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=score_collection_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_array_collection_callback(
        cls, array_collection_type: ArrayCollectionTypeT, resource_spec: ResourceSpecProtocolT
    ) -> ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_collection_registry()
        artifact = registry.get(artifact_type=array_collection_type, resource_spec=resource_spec)
        callback = ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=array_collection_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_plot_collection_callback(
        cls, plot_collection_type: PlotCollectionTypeT, resource_spec: ResourceSpecProtocolT
    ) -> ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_collection_registry()
        artifact = registry.get(artifact_type=plot_collection_type, resource_spec=resource_spec)
        callback = ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=plot_collection_type.name, artifact=artifact
        )
        return callback
