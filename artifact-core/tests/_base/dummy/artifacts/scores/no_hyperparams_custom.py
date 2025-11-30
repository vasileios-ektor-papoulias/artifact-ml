from artifact_core._base.core.hyperparams import NoArtifactHyperparams

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.registries.scores import DummyScoreRegistry
from tests._base.dummy.resources import DummyArtifactResources


@DummyScoreRegistry.register_custom_artifact(artifact_type="NO_HYPERPARAMS_CUSTOM_SCORE_ARTIFACT")
class NoHyperparamsCustomScoreArtifact(DummyArtifact[NoArtifactHyperparams, float]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        return 0
