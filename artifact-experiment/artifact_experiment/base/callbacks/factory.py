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

artifactTypeT = TypeVar("artifactTypeT", bound=ArtifactType)
scoreTypeT = TypeVar("scoreTypeT", bound=ArtifactType)
arrayTypeT = TypeVar("arrayTypeT", bound=ArtifactType)
plotTypeT = TypeVar("plotTypeT", bound=ArtifactType)
scoreCollectionTypeT = TypeVar("scoreCollectionTypeT", bound=ArtifactType)
arrayCollectionTypeT = TypeVar("arrayCollectionTypeT", bound=ArtifactType)
plotCollectionTypeT = TypeVar("plotCollectionTypeT", bound=ArtifactType)
resourceSpecProtocolT = TypeVar("resourceSpecProtocolT", bound=ResourceSpecProtocol)
artifactResourcesT = TypeVar("artifactResourcesT", bound=ArtifactResources)
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class ArtifactCallbackFactory(
    Generic[
        scoreTypeT,
        arrayTypeT,
        plotTypeT,
        scoreCollectionTypeT,
        arrayCollectionTypeT,
        plotCollectionTypeT,
        artifactResourcesT,
        resourceSpecProtocolT,
    ]
):
    @staticmethod
    @abstractmethod
    def _get_score_registry() -> Type[
        ArtifactRegistry[scoreTypeT, artifactResourcesT, float, resourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_registry() -> Type[
        ArtifactRegistry[arrayTypeT, artifactResourcesT, ndarray, resourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_registry() -> Type[
        ArtifactRegistry[plotTypeT, artifactResourcesT, Figure, resourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_registry() -> Type[
        ArtifactRegistry[
            scoreCollectionTypeT, artifactResourcesT, Dict[str, float], resourceSpecProtocolT
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_registry() -> Type[
        ArtifactRegistry[
            arrayCollectionTypeT, artifactResourcesT, Dict[str, ndarray], resourceSpecProtocolT
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_registry() -> Type[
        ArtifactRegistry[
            plotCollectionTypeT, artifactResourcesT, Dict[str, Figure], resourceSpecProtocolT
        ]
    ]: ...

    @classmethod
    def build_score_callback(
        cls, score_type: scoreTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactScoreCallback[artifactResourcesT, resourceSpecProtocolT]:
        registry = cls._get_score_registry()
        artifact = registry.get(artifact_type=score_type, resource_spec=resource_spec)
        callback = ArtifactScoreCallback[artifactResourcesT, resourceSpecProtocolT](
            key=score_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_array_callback(
        cls, array_type: arrayTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactArrayCallback[artifactResourcesT, resourceSpecProtocolT]:
        registry = cls._get_array_registry()
        artifact = registry.get(artifact_type=array_type, resource_spec=resource_spec)
        callback = ArtifactArrayCallback[artifactResourcesT, resourceSpecProtocolT](
            key=array_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_plot_callback(
        cls, plot_type: plotTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactPlotCallback[artifactResourcesT, resourceSpecProtocolT]:
        registry = cls._get_plot_registry()
        artifact = registry.get(artifact_type=plot_type, resource_spec=resource_spec)
        callback = ArtifactPlotCallback[artifactResourcesT, resourceSpecProtocolT](
            key=plot_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_score_collection_callback(
        cls, score_collection_type: scoreCollectionTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactScoreCollectionCallback[artifactResourcesT, resourceSpecProtocolT]:
        registry = cls._get_score_collection_registry()
        artifact = registry.get(artifact_type=score_collection_type, resource_spec=resource_spec)
        callback = ArtifactScoreCollectionCallback[artifactResourcesT, resourceSpecProtocolT](
            key=score_collection_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_array_collection_callback(
        cls, array_collection_type: arrayCollectionTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactArrayCollectionCallback[artifactResourcesT, resourceSpecProtocolT]:
        registry = cls._get_array_collection_registry()
        artifact = registry.get(artifact_type=array_collection_type, resource_spec=resource_spec)
        callback = ArtifactArrayCollectionCallback[artifactResourcesT, resourceSpecProtocolT](
            key=array_collection_type.name, artifact=artifact
        )
        return callback

    @classmethod
    def build_plot_collection_callback(
        cls, plot_collection_type: plotCollectionTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactPlotCollectionCallback[artifactResourcesT, resourceSpecProtocolT]:
        registry = cls._get_plot_collection_registry()
        artifact = registry.get(artifact_type=plot_collection_type, resource_spec=resource_spec)
        callback = ArtifactPlotCollectionCallback[artifactResourcesT, resourceSpecProtocolT](
            key=plot_collection_type.name, artifact=artifact
        )
        return callback
