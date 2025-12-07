from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar, Union

from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._base.orchestration.registry import ArtifactRegistry, ArtifactType
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactEngineT = TypeVar("ArtifactEngineT", bound="ArtifactEngine")


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
    def __init__(
        self,
        resource_spec: ResourceSpecProtocolT,
        score_registry: Type[
            ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ScoreTypeT, Score]
        ],
        array_registry: Type[
            ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArrayTypeT, Array]
        ],
        plot_registry: Type[
            ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, PlotTypeT, Plot]
        ],
        score_collection_registry: Type[
            ArtifactRegistry[
                ArtifactResourcesT, ResourceSpecProtocolT, ScoreCollectionTypeT, ScoreCollection
            ]
        ],
        array_collection_registry: Type[
            ArtifactRegistry[
                ArtifactResourcesT, ResourceSpecProtocolT, ArrayCollectionTypeT, ArrayCollection
            ]
        ],
        plot_collection_registry: Type[
            ArtifactRegistry[
                ArtifactResourcesT, ResourceSpecProtocolT, PlotCollectionTypeT, PlotCollection
            ]
        ],
    ):
        self._resource_spec = resource_spec
        self._score_registry = score_registry
        self._array_registry = array_registry
        self._plot_registry = plot_registry
        self._score_collection_registry = score_collection_registry
        self._array_collection_registry = array_collection_registry
        self._plot_collection_registry = plot_collection_registry

    @classmethod
    def build(cls: Type[ArtifactEngineT], resource_spec: ResourceSpecProtocolT) -> ArtifactEngineT:
        return cls(
            resource_spec=resource_spec,
            score_registry=cls._get_score_registry(),
            array_registry=cls._get_array_registry(),
            plot_registry=cls._get_plot_registry(),
            score_collection_registry=cls._get_score_collection_registry(),
            array_collection_registry=cls._get_array_collection_registry(),
            plot_collection_registry=cls._get_plot_collection_registry(),
        )

    @property
    def resource_spec(self) -> ResourceSpecProtocolT:
        return self._resource_spec

    @classmethod
    @abstractmethod
    def _get_score_registry(
        cls,
    ) -> Type[ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ScoreTypeT, Score]]: ...

    @classmethod
    @abstractmethod
    def _get_array_registry(
        cls,
    ) -> Type[ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArrayTypeT, Array]]: ...

    @classmethod
    @abstractmethod
    def _get_plot_registry(
        cls,
    ) -> Type[ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, PlotTypeT, Plot]]: ...

    @classmethod
    @abstractmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, ScoreCollectionTypeT, ScoreCollection
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, ArrayCollectionTypeT, ArrayCollection
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, PlotCollectionTypeT, PlotCollection
        ]
    ]: ...

    def produce_score(
        self, score_type: Union[ScoreTypeT, str], resources: ArtifactResourcesT
    ) -> Score:
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

    def produce_plot(self, plot_type: Union[PlotTypeT, str], resources: ArtifactResourcesT) -> Plot:
        artifact = self._plot_registry.get(
            artifact_type=plot_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_score_collection(
        self,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        resources: ArtifactResourcesT,
    ) -> ScoreCollection:
        artifact = self._score_collection_registry.get(
            artifact_type=score_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_array_collection(
        self,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        resources: ArtifactResourcesT,
    ) -> ArrayCollection:
        artifact = self._array_collection_registry.get(
            artifact_type=array_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)

    def produce_plot_collection(
        self,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        resources: ArtifactResourcesT,
    ) -> PlotCollection:
        artifact = self._plot_collection_registry.get(
            artifact_type=plot_collection_type, resource_spec=self._resource_spec
        )
        return artifact.compute(resources=resources)
