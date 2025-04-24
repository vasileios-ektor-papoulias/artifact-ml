from abc import ABC, abstractmethod
from typing import Dict, Generic, Type, TypeVar

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType

scoreTypeT = TypeVar("scoreTypeT", bound="ArtifactType")
arrayTypeT = TypeVar("arrayTypeT", bound="ArtifactType")
plotTypeT = TypeVar("plotTypeT", bound="ArtifactType")
scoreCollectionTypeT = TypeVar("scoreCollectionTypeT", bound="ArtifactType")
arrayCollectionTypeT = TypeVar("arrayCollectionTypeT", bound="ArtifactType")
plotCollectionTypeT = TypeVar("plotCollectionTypeT", bound="ArtifactType")
artifactResourcesT = TypeVar("artifactResourcesT", bound="ArtifactResources")
resourceSpecProtocolT = TypeVar("resourceSpecProtocolT", bound=ResourceSpecProtocol)


class ArtifactEngine(
    ABC,
    Generic[
        artifactResourcesT,
        resourceSpecProtocolT,
        scoreTypeT,
        arrayTypeT,
        plotTypeT,
        scoreCollectionTypeT,
        arrayCollectionTypeT,
        plotCollectionTypeT,
    ],
):
    def __init__(self, resource_spec: resourceSpecProtocolT):
        self._resource_spec = resource_spec
        self._score_registry = self._get_score_registry()
        self._array_registry = self._get_array_registry()
        self._plot_registry = self._get_plot_registry()
        self._score_collection_registry = self._get_score_collection_registry()
        self._array_collection_registry = self._get_array_collection_registry()
        self._plot_collection_registry = self._get_plot_collection_registry()

    @property
    def resource_spec(self) -> resourceSpecProtocolT:
        return self._resource_spec

    @property
    def score_registry(
        self,
    ) -> Type[ArtifactRegistry[scoreTypeT, artifactResourcesT, float, resourceSpecProtocolT]]:
        return self._score_registry

    @property
    def array_registry(
        self,
    ) -> Type[ArtifactRegistry[arrayTypeT, artifactResourcesT, ndarray, resourceSpecProtocolT]]:
        return self._array_registry

    @property
    def plot_registry(
        self,
    ) -> Type[ArtifactRegistry[plotTypeT, artifactResourcesT, Figure, resourceSpecProtocolT]]:
        return self._plot_registry

    @property
    def score_collection_registry(
        self,
    ) -> Type[
        ArtifactRegistry[
            scoreCollectionTypeT, artifactResourcesT, Dict[str, float], resourceSpecProtocolT
        ]
    ]:
        return self._score_collection_registry

    @property
    def array_collection_registry(
        self,
    ) -> Type[
        ArtifactRegistry[
            arrayCollectionTypeT, artifactResourcesT, Dict[str, ndarray], resourceSpecProtocolT
        ]
    ]:
        return self._array_collection_registry

    @property
    def plot_collection_registry(
        self,
    ) -> Type[
        ArtifactRegistry[
            plotCollectionTypeT, artifactResourcesT, Dict[str, Figure], resourceSpecProtocolT
        ]
    ]:
        return self._plot_collection_registry

    @classmethod
    @abstractmethod
    def _get_score_registry(
        cls,
    ) -> Type[ArtifactRegistry[scoreTypeT, artifactResourcesT, float, resourceSpecProtocolT]]: ...

    @classmethod
    @abstractmethod
    def _get_array_registry(
        cls,
    ) -> Type[ArtifactRegistry[arrayTypeT, artifactResourcesT, ndarray, resourceSpecProtocolT]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_registry(
        cls,
    ) -> Type[ArtifactRegistry[plotTypeT, artifactResourcesT, Figure, resourceSpecProtocolT]]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            scoreCollectionTypeT,
            artifactResourcesT,
            Dict[str, float],
            resourceSpecProtocolT,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            arrayCollectionTypeT,
            artifactResourcesT,
            Dict[str, ndarray],
            resourceSpecProtocolT,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            plotCollectionTypeT,
            artifactResourcesT,
            Dict[str, Figure],
            resourceSpecProtocolT,
        ]
    ]: ...

    def produce_score(self, score_type: scoreTypeT, resources: artifactResourcesT) -> float:
        artifact = self._score_registry.get(
            artifact_type=score_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_array(self, array_type: arrayTypeT, resources: artifactResourcesT) -> ndarray:
        artifact = self._array_registry.get(
            artifact_type=array_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_plot(self, plot_type: plotTypeT, resources: artifactResourcesT) -> Figure:
        artifact = self._plot_registry.get(
            artifact_type=plot_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_score_collection(
        self,
        score_collection_type: scoreCollectionTypeT,
        resources: artifactResourcesT,
    ) -> Dict[str, float]:
        artifact = self._score_collection_registry.get(
            artifact_type=score_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_array_collection(
        self,
        array_collection_type: arrayCollectionTypeT,
        resources: artifactResourcesT,
    ) -> Dict[str, ndarray]:
        artifact = self._array_collection_registry.get(
            artifact_type=array_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_plot_collection(
        self,
        plot_collection_type: plotCollectionTypeT,
        resources: artifactResourcesT,
    ) -> Dict[str, Figure]:
        artifact = self._plot_collection_registry.get(
            artifact_type=plot_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)
