from dataclasses import dataclass
from typing import Optional, TypeVar

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResult,
    NoArtifactHyperparams,
)

from tests.base.dummy.artifact_dependencies import DummyArtifactResources, DummyDataSpec
from tests.base.dummy.registries import (
    AlternativeDummyScoreRegistry,
    DummyScoreRegistry,
    DummyScoreType,
    InvalidParamDummyScoreRegistry,
    MissingParamDummyScoreRegistry,
)

artifactHyperparamsT = TypeVar("artifactHyperparamsT", bound=ArtifactHyperparams)
artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class DummyArtifact(
    Artifact[DummyArtifactResources, artifactResultT, artifactHyperparamsT, DummyDataSpec]
):
    def __init__(
        self, data_spec: DummyDataSpec, hyperparams: Optional[artifactHyperparamsT] = None
    ):
        self._data_spec = data_spec
        self._hyperparams = hyperparams


@InvalidParamDummyScoreRegistry.register_artifact_config(
    artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT
)
@MissingParamDummyScoreRegistry.register_artifact_config(
    artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT
)
@AlternativeDummyScoreRegistry.register_artifact_config(
    artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT
)
@DummyScoreRegistry.register_artifact_config(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@dataclass(frozen=True)
class DummyScoreHyperparams(ArtifactHyperparams):
    adjust_scale: bool


@InvalidParamDummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@MissingParamDummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@AlternativeDummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@DummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
class DummyScoreArtifact(DummyArtifact[float, DummyScoreHyperparams]):
    def __init__(self, data_spec: DummyDataSpec, hyperparams: DummyScoreHyperparams):
        self._data_spec = data_spec
        self._hyperparams = hyperparams

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        result = resources.x
        if self._hyperparams.adjust_scale:
            result = result * self._data_spec.scale
        return result


@DummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.NO_HYPERPARAMS_ARTIFACT)
class NoHyperparamsArtifact(DummyArtifact[float, NoArtifactHyperparams]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        result = resources.x * self._data_spec.scale
        return result


@AlternativeDummyScoreRegistry.register_artifact(
    artifact_type=DummyScoreType.IN_ALTERNATIVE_REGISTRY
)
class AlternativeRegistryArtifact(DummyArtifact[float, NoArtifactHyperparams]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        result = resources.x * self._data_spec.scale
        return result
