from typing import Any, Dict, Generic, Mapping, Optional, Type, TypeVar, Union

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import (
    NO_ARTIFACT_HYPERPARAMS,
    ArtifactHyperparams,
    NoArtifactHyperparams,
)
from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._base.orchestration.key_formatter import ArtifactKeyFormatter
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import (
    ArtifactResult,
)

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)

ArtifactCollection = Dict[
    str,
    Type[Artifact[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparams, ArtifactResultT]],
]
ArtifactHyperparamsCollection = Dict[str, Type[ArtifactHyperparams]]
Registree = Union[Artifact[Any, Any, Any, Any], ArtifactHyperparams]
RegistreeT = TypeVar("RegistreeT", bound=Registree)
ArtifactT = TypeVar("ArtifactT", bound=Artifact[Any, Any, Any, Any])
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)


class ArtifactRegistryReader(
    Generic[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT]
):
    @classmethod
    def get(
        cls,
        artifact_type: Union[ArtifactTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        artifact_registry: ArtifactCollection,
        artifact_hyperparams_registry: ArtifactHyperparamsCollection,
        artifact_configurations: Mapping[str, Mapping[str, Any]],
    ) -> Artifact[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparams, ArtifactResultT]:
        key = ArtifactKeyFormatter.get_artifact_key(artifact_type=artifact_type)
        return cls._get(
            key=key,
            resource_spec=resource_spec,
            artifact_registry=artifact_registry,
            artifact_hyperparams_registry=artifact_hyperparams_registry,
            artifact_configurations=artifact_configurations,
        )

    @classmethod
    def _get(
        cls,
        key: str,
        resource_spec: ResourceSpecProtocolT,
        artifact_registry: ArtifactCollection,
        artifact_hyperparams_registry: ArtifactHyperparamsCollection,
        artifact_configurations: Mapping[str, Mapping[str, Any]],
    ) -> Artifact[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparams, ArtifactResultT]:
        artifact_class = cls._get_artifact_class(key=key, registry=artifact_registry)
        artifact_config = cls._get_artifact_config(
            key=key, artifact_configurations=artifact_configurations
        )
        artifact_hyperparams_class = cls._get_artifact_hyperparams_class(
            key=key, registry=artifact_hyperparams_registry
        )
        hyperparams = cls._build_artifact_hyperparams(
            artifact_hyperparams_class=artifact_hyperparams_class, artifact_config=artifact_config
        )
        artifact = artifact_class(resource_spec=resource_spec, hyperparams=hyperparams)
        return artifact

    @classmethod
    def _get_artifact_class(
        cls, key: str, registry: ArtifactCollection
    ) -> Type[
        Artifact[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparams, ArtifactResultT]
    ]:
        artifact_class = registry.get(key, None)
        if artifact_class is None:
            raise ValueError(f"Artifact {key} not registered")
        return artifact_class

    @classmethod
    def _get_artifact_hyperparams_class(
        cls, key: str, registry: ArtifactHyperparamsCollection
    ) -> Type[ArtifactHyperparams]:
        artifact_hyperparams_class = registry.get(key, NoArtifactHyperparams)
        return artifact_hyperparams_class

    @classmethod
    def _get_artifact_config(
        cls, key: str, artifact_configurations: Mapping[str, Mapping[str, Any]]
    ) -> Optional[Mapping[str, Any]]:
        artifact_config = artifact_configurations.get(key, None)
        return artifact_config

    @classmethod
    def _build_artifact_hyperparams(
        cls,
        artifact_hyperparams_class: Type[ArtifactHyperparams],
        artifact_config: Optional[Mapping[str, Any]],
    ) -> ArtifactHyperparams:
        if artifact_hyperparams_class == NoArtifactHyperparams:
            artifact_hyperparams = NO_ARTIFACT_HYPERPARAMS
        elif artifact_config is None:
            raise ValueError(
                f"Missing config for hyperparams type {artifact_hyperparams_class.__name__}"
            )
        else:
            try:
                artifact_hyperparams = artifact_hyperparams_class.build(**artifact_config)
            except TypeError as e:
                raise ValueError(
                    f"Error instantiating '{artifact_hyperparams_class.__name__}'"
                    f"with arguments {artifact_config}: {e}"
                ) from e
        return artifact_hyperparams
