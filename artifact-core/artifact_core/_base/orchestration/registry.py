from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, Mapping, Type, TypeVar, Union

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._base.orchestration.registry_reader import ArtifactRegistryReader
from artifact_core._base.orchestration.registry_writer import ArtifactRegistryWriter
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

ArtifactT = TypeVar("ArtifactT", bound=Artifact[Any, Any, Any, Any])
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactRegistry(
    Generic[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT]
):
    _artifact_registry: Dict[
        str,
        Type[
            Artifact[
                ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparams, ArtifactResultT
            ]
        ],
    ] = {}
    _artifact_hyperparams_registry: Dict[str, Type[ArtifactHyperparams]] = {}
    _artifact_configurations = {}

    @classmethod
    @abstractmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._artifact_registry = {}
        cls._artifact_hyperparams_registry = {}

    @classmethod
    def get(
        cls, artifact_type: Union[ArtifactTypeT, str], resource_spec: ResourceSpecProtocolT
    ) -> Artifact[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparams, ArtifactResultT]:
        if not cls._artifact_configurations:
            cls._artifact_configurations = cls._get_artifact_configurations()
        return ArtifactRegistryReader[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].get(
            artifact_type=artifact_type,
            resource_spec=resource_spec,
            artifact_registry=cls._artifact_registry,
            artifact_hyperparams_registry=cls._artifact_hyperparams_registry,
            artifact_configurations=cls._artifact_configurations,
        )

    @classmethod
    def register_artifact(
        cls, artifact_type: ArtifactTypeT
    ) -> Callable[[Type[ArtifactT]], Type[ArtifactT]]:
        return ArtifactRegistryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].register_artifact(artifact_type=artifact_type, registry=cls._artifact_registry)

    @classmethod
    def register_artifact_hyperparams(
        cls, artifact_type: ArtifactTypeT
    ) -> Callable[[Type[ArtifactHyperparamsT]], Type[ArtifactHyperparamsT]]:
        return ArtifactRegistryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].register_artifact_hyperparams(
            artifact_type=artifact_type, registry=cls._artifact_hyperparams_registry
        )

    @classmethod
    def register_custom_artifact(
        cls, artifact_type: str
    ) -> Callable[[Type[ArtifactT]], Type[ArtifactT]]:
        return ArtifactRegistryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].register_artifact(artifact_type=artifact_type, registry=cls._artifact_registry)

    @classmethod
    def register_custom_artifact_hyperparams(
        cls, artifact_type: str
    ) -> Callable[[Type[ArtifactHyperparamsT]], Type[ArtifactHyperparamsT]]:
        return ArtifactRegistryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].register_artifact_hyperparams(
            artifact_type=artifact_type, registry=cls._artifact_hyperparams_registry
        )


ScoreRegistry = ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, Score]
ArrayRegistry = ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, Array]
PlotRegistry = ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, Plot]
ScoreCollectionRegistry = ArtifactRegistry[
    ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ScoreCollection
]
ArrayCollectionRegistry = ArtifactRegistry[
    ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArrayCollection
]
PlotCollectionRegistry = ArtifactRegistry[
    ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, PlotCollection
]
