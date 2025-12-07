from artifact_core._base.core.hyperparams import ArtifactHyperparams

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.resources import DummyArtifactResources


class UnregisteredArtifactHyperparams(ArtifactHyperparams):
    test_param: int = 1


class UnregisteredArtifact(DummyArtifact[UnregisteredArtifactHyperparams, float]):
    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        return 0.0
