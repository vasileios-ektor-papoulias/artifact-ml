from dataclasses import dataclass

from artifact_core._base.core.hyperparams import ArtifactHyperparams

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.registries.scores import (
    AlternativeDummyScoreRegistry,
    DummyScoreRegistry,
    InvalidParamDummyScoreRegistry,
    MissingParamDummyScoreRegistry,
)
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources
from tests._base.dummy.types.scores import DummyScoreType


@InvalidParamDummyScoreRegistry.register_artifact_hyperparams(
    artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT
)
@MissingParamDummyScoreRegistry.register_artifact_hyperparams(
    artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT
)
@AlternativeDummyScoreRegistry.register_artifact_hyperparams(
    artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT
)
@DummyScoreRegistry.register_artifact_hyperparams(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@dataclass(frozen=True)
class DummyScoreHyperparams(ArtifactHyperparams):
    adjust_scale: bool


@InvalidParamDummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@MissingParamDummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@AlternativeDummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
@DummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)
class DummyScoreArtifact(DummyArtifact[DummyScoreHyperparams, float]):
    def __init__(self, resource_spec: DummyResourceSpec, hyperparams: DummyScoreHyperparams):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        result = resources.x
        if self._hyperparams.adjust_scale:
            result = result * self._resource_spec.scale
        return result
