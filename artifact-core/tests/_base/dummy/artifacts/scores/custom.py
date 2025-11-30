from dataclasses import dataclass

from artifact_core._base.core.hyperparams import ArtifactHyperparams

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.registries.scores import DummyScoreRegistry
from tests._base.dummy.resources import DummyArtifactResources


@DummyScoreRegistry.register_custom_artifact_hyperparams(artifact_type="CUSTOM_SCORE_ARTIFACT")
@dataclass(frozen=True)
class CustomScoreHyperparams(ArtifactHyperparams):
    result: float


@DummyScoreRegistry.register_custom_artifact(artifact_type="CUSTOM_SCORE_ARTIFACT")
class CustomScoreArtifact(DummyArtifact[CustomScoreHyperparams, float]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        assert self._hyperparams is not None
        return self._hyperparams.result
