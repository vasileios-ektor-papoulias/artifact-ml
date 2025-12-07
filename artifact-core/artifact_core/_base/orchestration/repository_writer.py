import warnings
from typing import Any, Callable, Dict, Generic, Type, TypeVar, Union

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._base.orchestration.key_formatter import ArtifactKeyFormatter
from artifact_core._base.orchestration.repository import (
    ArtifactHyperparamsRepository,
    ArtifactRepository,
)
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult

Registree = Union[Artifact[Any, Any, Any, Any], ArtifactHyperparams]
RegistreeT = TypeVar("RegistreeT", bound=Registree)
ArtifactT = TypeVar("ArtifactT", bound=Artifact[Any, Any, Any, Any])
ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactRepositoryWriter(
    Generic[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactTypeT, ArtifactResultT]
):
    @classmethod
    def put_artifact(
        cls,
        artifact_type: Union[ArtifactTypeT, str],
        repository: ArtifactRepository[ArtifactResourcesT, ResourceSpecProtocolT, ArtifactResultT],
    ) -> Callable[[Type[ArtifactT]], Type[ArtifactT]]:
        key = ArtifactKeyFormatter.get_artifact_key(artifact_type=artifact_type)
        return cls._put(
            key=key,
            repository=repository,
            warning_message=f"Artifact already registered for artifact_type={key}",
        )

    @classmethod
    def put_artifact_hyperparams(
        cls, artifact_type: Union[ArtifactTypeT, str], repository: ArtifactHyperparamsRepository
    ) -> Callable[[Type[ArtifactHyperparamsT]], Type[ArtifactHyperparamsT]]:
        key = ArtifactKeyFormatter.get_artifact_key(artifact_type=artifact_type)
        return cls._put(
            key=key,
            repository=repository,
            warning_message=f"Hyperparams already registered for artifact_type={key}",
        )

    @staticmethod
    def _put(
        key: str,
        repository: Dict[str, Any],
        warning_message: str,
    ) -> Callable[[Type[RegistreeT]], Type[RegistreeT]]:
        def insertion_decorator(item: Type[RegistreeT]) -> Type[RegistreeT]:
            if key not in repository:
                repository[key] = item
            else:
                warnings.warn(warning_message, UserWarning, stacklevel=3)
            return item

        return insertion_decorator
