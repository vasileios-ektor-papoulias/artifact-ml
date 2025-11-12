from abc import ABC, abstractmethod
from typing import Dict, Generic, Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core._base.contracts.resource_spec import ResourceSpecProtocol
from artifact_core._base.contracts.resources import ArtifactResources
from artifact_core._base.orchestration.registry import ArtifactRegistry, ArtifactType
from artifact_core._base.types.artifact_result import Array

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
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

    @classmethod
    @abstractmethod
    def _get_score_registry(
        cls,
    ) -> Type[ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ScoreTypeT, float]]: ...

    @classmethod
    @abstractmethod
    def _get_array_registry(
        cls,
    ) -> Type[ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArrayTypeT, Array]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_registry(
        cls,
    ) -> Type[ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, PlotTypeT, Figure]]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, ScoreCollectionTypeT, Dict[str, float]
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, ArrayCollectionTypeT, Dict[str, Array]
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, PlotCollectionTypeT, Dict[str, Figure]
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
    ) -> Array:
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
    ) -> Dict[str, Array]:
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
