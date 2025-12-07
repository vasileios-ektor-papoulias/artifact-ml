from typing import TypeVar

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.classification.artifact import ClassificationArtifact

from tests._domains.classification.dummy.resource_spec import DummyClassSpec
from tests._domains.classification.dummy.resources import (
    DummyClassificationResults,
    DummyClassStore,
)

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)

DummyClassificationArtifact = ClassificationArtifact[
    DummyClassStore,
    DummyClassificationResults,
    DummyClassSpec,
    ArtifactHyperparamsT,
    ArtifactResultT,
]
