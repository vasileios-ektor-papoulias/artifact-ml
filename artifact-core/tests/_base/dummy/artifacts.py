from dataclasses import dataclass
from typing import Optional, TypeVar

from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import ArtifactHyperparams, NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArtifactResult

from tests._base.dummy.registries import (
    AlternativeDummyScoreRegistry,
    DummyScoreRegistry,
    DummyScoreType,
    InvalidParamDummyScoreRegistry,
    MissingParamDummyScoreRegistry,
)
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound=ArtifactHyperparams)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


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


@DummyScoreRegistry.register_custom_artifact_hyperparams(artifact_type="CUSTOM_SCORE_ARTIFACT")
@dataclass(frozen=True)
class CustomScoreHyperparams(ArtifactHyperparams):
    result: float


class UnregisteredArtifactHyperparams(ArtifactHyperparams):
    test_param: int = 1


class DummyArtifact(
    Artifact[DummyArtifactResources, DummyResourceSpec, ArtifactHyperparamsT, ArtifactResultT]
):
    def __init__(
        self, resource_spec: DummyResourceSpec, hyperparams: Optional[ArtifactHyperparamsT] = None
    ):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams


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


@DummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.NO_HYPERPARAMS_ARTIFACT)
class NoHyperparamsArtifact(DummyArtifact[NoArtifactHyperparams, float]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        result = resources.x * self._resource_spec.scale
        return result


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


@DummyScoreRegistry.register_custom_artifact(artifact_type="NO_HYPERPARAMS_CUSTOM_SCORE_ARTIFACT")
class NoHyperparamsCustomScoreArtifact(DummyArtifact[NoArtifactHyperparams, float]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        if not resources.valid:
            raise ValueError("Invalid Resources")
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        return 0


class UnregisteredArtifact(DummyArtifact[UnregisteredArtifactHyperparams, float]):
    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        return 0.0
