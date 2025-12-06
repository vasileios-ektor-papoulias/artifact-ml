from typing import TypeVar

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArtifactResult
from artifact_core._domains.classification.artifact import ClassificationArtifact
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)

DummyClassificationArtifact = ClassificationArtifact[
    ClassStore,
    ClassificationResults,
    ClassSpec,
    ArtifactHyperparamsT,
    ArtifactResultT,
]
