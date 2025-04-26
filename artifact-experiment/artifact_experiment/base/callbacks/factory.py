from abc import abstractmethod
from typing import Dict, Generic, Type, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_experiment.base.callbacks.artifact import ArtifactCallback
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
    ) -> ArtifactCallback[artifactResourcesT, float, resourceSpecProtocolT]:
        registry = cls._get_score_registry()
        callback = cls._build_artifact_callback(
            artifact_type=score_type, resource_spec=resource_spec, registry=registry
        )
        return callback

    @classmethod
    def build_array_callback(
        cls, array_type: arrayTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactCallback[artifactResourcesT, ndarray, resourceSpecProtocolT]:
        registry = cls._get_array_registry()
        callback = cls._build_artifact_callback(
            artifact_type=array_type, resource_spec=resource_spec, registry=registry
        )
        return callback

    @classmethod
    def build_plot_callback(
        cls, plot_type: plotTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactCallback[artifactResourcesT, Figure, resourceSpecProtocolT]:
        registry = cls._get_plot_registry()
        callback = cls._build_artifact_callback(
            artifact_type=plot_type, resource_spec=resource_spec, registry=registry
        )
        return callback

    @classmethod
    def build_score_collection_callback(
        cls, score_collection_type: scoreCollectionTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactCallback[artifactResourcesT, Dict[str, float], resourceSpecProtocolT]:
        registry = cls._get_score_collection_registry()
        callback = cls._build_artifact_callback(
            artifact_type=score_collection_type, resource_spec=resource_spec, registry=registry
        )
        return callback

    @classmethod
    def build_array_collection_callback(
        cls, array_collection_type: arrayCollectionTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactCallback[artifactResourcesT, Dict[str, ndarray], resourceSpecProtocolT]:
        registry = cls._get_array_collection_registry()
        callback = cls._build_artifact_callback(
            artifact_type=array_collection_type, resource_spec=resource_spec, registry=registry
        )
        return callback

    @classmethod
    def build_plot_collection_callback(
        cls, plot_collection_type: plotCollectionTypeT, resource_spec: resourceSpecProtocolT
    ) -> ArtifactCallback[artifactResourcesT, Dict[str, Figure], resourceSpecProtocolT]:
        registry = cls._get_plot_collection_registry()
        callback = cls._build_artifact_callback(
            artifact_type=plot_collection_type, resource_spec=resource_spec, registry=registry
        )
        return callback

    @staticmethod
    def _build_artifact_callback(
        artifact_type: artifactTypeT,
        resource_spec: resourceSpecProtocolT,
        registry: Type[
            ArtifactRegistry[
                artifactTypeT, artifactResourcesT, artifactResultT, resourceSpecProtocolT
            ]
        ],
    ) -> ArtifactCallback[artifactResourcesT, artifactResultT, resourceSpecProtocolT]:
        artifact = registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
        callback = ArtifactCallback[artifactResourcesT, artifactResultT, resourceSpecProtocolT](
            key=artifact_type.name, artifact=artifact
        )
        return callback
