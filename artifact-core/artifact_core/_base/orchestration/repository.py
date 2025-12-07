from typing import Any, Dict, Type, TypeVar

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.core.resource_spec import ResourceSpecProtocol
from artifact_core._base.core.resources import ArtifactResources
from artifact_core._base.primitives.artifact_type import ArtifactType
from artifact_core._base.typing.artifact_result import ArtifactResult

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


ArtifactRepository = Dict[
    str,
    Type[Artifact[ArtifactResourcesT, ResourceSpecProtocolT, Any, ArtifactResultT]],
]
ArtifactHyperparamsRepository = Dict[str, Type[ArtifactHyperparams]]
