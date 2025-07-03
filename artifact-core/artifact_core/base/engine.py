from abc import ABC, abstractmethod
from typing import Dict, Generic, Type, TypeVar, Union

from matplotlib.figure import Figure
from numpy import ndarray

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType

ScoreTypeT = TypeVar("ScoreTypeT", bound="ArtifactType")
ArrayTypeT = TypeVar("ArrayTypeT", bound="ArtifactType")
PlotTypeT = TypeVar("PlotTypeT", bound="ArtifactType")
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound="ArtifactType")
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound="ArtifactType")
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound="ArtifactType")
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound="ArtifactResources")
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)


class ArtifactEngine(
    ABC,
    Generic[
        ArtifactResourcesT,
        ResourceSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
):
    def __init__(self, resource_spec: ResourceSpecProtocolT):
        self._resource_spec = resource_spec
        self._score_registry = self._get_score_registry()
        self._array_registry = self._get_array_registry()
        self._plot_registry = self._get_plot_registry()
        self._score_collection_registry = self._get_score_collection_registry()
        self._array_collection_registry = self._get_array_collection_registry()
        self._plot_collection_registry = self._get_plot_collection_registry()

    @property
    def resource_spec(self) -> ResourceSpecProtocolT:
        return self._resource_spec

    @property
    def score_registry(
        self,
    ) -> Type[ArtifactRegistry[ScoreTypeT, ArtifactResourcesT, float, ResourceSpecProtocolT]]:
        return self._score_registry

    @property
    def array_registry(
        self,
    ) -> Type[ArtifactRegistry[ArrayTypeT, ArtifactResourcesT, ndarray, ResourceSpecProtocolT]]:
        return self._array_registry

    @property
    def plot_registry(
        self,
    ) -> Type[ArtifactRegistry[PlotTypeT, ArtifactResourcesT, Figure, ResourceSpecProtocolT]]:
        return self._plot_registry

    @property
    def score_collection_registry(
        self,
    ) -> Type[
        ArtifactRegistry[
            ScoreCollectionTypeT, ArtifactResourcesT, Dict[str, float], ResourceSpecProtocolT
        ]
    ]:
        return self._score_collection_registry

    @property
    def array_collection_registry(
        self,
    ) -> Type[
        ArtifactRegistry[
            ArrayCollectionTypeT, ArtifactResourcesT, Dict[str, ndarray], ResourceSpecProtocolT
        ]
    ]:
        return self._array_collection_registry

    @property
    def plot_collection_registry(
        self,
    ) -> Type[
        ArtifactRegistry[
            PlotCollectionTypeT, ArtifactResourcesT, Dict[str, Figure], ResourceSpecProtocolT
        ]
    ]:
        return self._plot_collection_registry

    @classmethod
    @abstractmethod
    def _get_score_registry(
        cls,
    ) -> Type[ArtifactRegistry[ScoreTypeT, ArtifactResourcesT, float, ResourceSpecProtocolT]]: ...

    @classmethod
    @abstractmethod
    def _get_array_registry(
        cls,
    ) -> Type[ArtifactRegistry[ArrayTypeT, ArtifactResourcesT, ndarray, ResourceSpecProtocolT]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_registry(
        cls,
    ) -> Type[ArtifactRegistry[PlotTypeT, ArtifactResourcesT, Figure, ResourceSpecProtocolT]]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ScoreCollectionTypeT,
            ArtifactResourcesT,
            Dict[str, float],
            ResourceSpecProtocolT,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ArrayCollectionTypeT,
            ArtifactResourcesT,
            Dict[str, ndarray],
            ResourceSpecProtocolT,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            PlotCollectionTypeT,
            ArtifactResourcesT,
            Dict[str, Figure],
            ResourceSpecProtocolT,
        ]
    ]: ...

    def produce_score(
        self, score_type: Union[ScoreTypeT, str], resources: ArtifactResourcesT
    ) -> float:
        artifact = self._score_registry.get(
            artifact_type=score_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_array(
        self, array_type: Union[ArrayTypeT, str], resources: ArtifactResourcesT
    ) -> ndarray:
        artifact = self._array_registry.get(
            artifact_type=array_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_plot(
        self, plot_type: Union[PlotTypeT, str], resources: ArtifactResourcesT
    ) -> Figure:
        artifact = self._plot_registry.get(
            artifact_type=plot_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        resources: ArtifactResourcesT,
    ) -> Dict[str, float]:
        artifact = self._score_collection_registry.get(
            artifact_type=score_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        resources: ArtifactResourcesT,
    ) -> Dict[str, ndarray]:
        artifact = self._array_collection_registry.get(
            artifact_type=array_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        resources: ArtifactResourcesT,
    ) -> Dict[str, Figure]:
        artifact = self._plot_collection_registry.get(
            artifact_type=plot_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)
