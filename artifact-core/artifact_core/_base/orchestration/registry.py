from abc import abstractmethod
from typing import Any, Callable, Generic, Mapping, Type, TypeVar, Union

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._base.orchestration.repository import (
    ArtifactHyperparamsRepository,
    ArtifactRepository,
)
from artifact_core._base.orchestration.repository_reader import ArtifactRepositoryReader
from artifact_core._base.orchestration.repository_writer import ArtifactRepositoryWriter
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult

ArtifactT = TypeVar("ArtifactT", bound=Artifact[Any, Any, Any, Any])
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactRegistry(
    Generic[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT]
):
    _artifact_repository: ArtifactRepository[
        ArtifactResourcesT, ResourceSpecProtocolT, ArtifactResultT
    ] = {}
    _artifact_hyperparams_repository: ArtifactHyperparamsRepository = {}
    _artifact_configurations = {}

    @classmethod
    @abstractmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._artifact_repository = {}
        cls._artifact_hyperparams_repository = {}

    @classmethod
    def get(
        cls, artifact_type: Union[ArtifactTypeT, str], resource_spec: ResourceSpecProtocolT
    ) -> Artifact[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactHyperparams, ArtifactResultT]:
        if not cls._artifact_configurations:
            cls._artifact_configurations = cls._get_artifact_configurations()
        return ArtifactRepositoryReader[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].get(
            artifact_type=artifact_type,
            resource_spec=resource_spec,
            artifact_repository=cls._artifact_repository,
            artifact_hyperparams_repository=cls._artifact_hyperparams_repository,
            artifact_configurations=cls._artifact_configurations,
        )

    @classmethod
    def register_artifact(
        cls, artifact_type: ArtifactTypeT
    ) -> Callable[[Type[ArtifactT]], Type[ArtifactT]]:
        return ArtifactRepositoryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].put_artifact(artifact_type=artifact_type, repository=cls._artifact_repository)

    @classmethod
    def register_artifact_hyperparams(
        cls, artifact_type: ArtifactTypeT
    ) -> Callable[[Type[ArtifactHyperparamsT]], Type[ArtifactHyperparamsT]]:
        return ArtifactRepositoryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].put_artifact_hyperparams(
            artifact_type=artifact_type, repository=cls._artifact_hyperparams_repository
        )

    @classmethod
    def register_custom_artifact(
        cls, artifact_type: str
    ) -> Callable[[Type[ArtifactT]], Type[ArtifactT]]:
        return ArtifactRepositoryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].put_artifact(artifact_type=artifact_type, repository=cls._artifact_repository)

    @classmethod
    def register_custom_artifact_hyperparams(
        cls, artifact_type: str
    ) -> Callable[[Type[ArtifactHyperparamsT]], Type[ArtifactHyperparamsT]]:
        return ArtifactRepositoryWriter[
            ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT
        ].put_artifact_hyperparams(
            artifact_type=artifact_type, repository=cls._artifact_hyperparams_repository
        )
