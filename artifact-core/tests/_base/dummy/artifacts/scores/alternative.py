from artifact_core._base.core.hyperparams import NoArtifactHyperparams

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.registries.scores import AlternativeDummyScoreRegistry
from tests._base.dummy.resources import DummyArtifactResources
from tests._base.dummy.types.scores import DummyScoreType


@AlternativeDummyScoreRegistry.register_artifact(
    artifact_type=DummyScoreType.IN_ALTERNATIVE_REGISTRY
)
class AlternativeRegistryArtifact(DummyArtifact[NoArtifactHyperparams, float]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        result = resources.x * self._resource_spec.scale
        return result
