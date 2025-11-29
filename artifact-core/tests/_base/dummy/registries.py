from typing import Any, Dict

from artifact_core._base.orchestration.registry import (
    ArtifactRegistry,
    ArtifactType,
)
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources


class DummyScoreType(ArtifactType):
    DUMMY_SCORE_ARTIFACT = "dummy_score_artifact"
    NO_HYPERPARAMS_ARTIFACT = "no_hyperparams_artifact"
    IN_ALTERNATIVE_REGISTRY = "in_alternative_registry"
    NOT_REGISTERED = "not_registered"


class DummyScoreRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyScoreType, Score]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {"adjust_scale": True},
            "CUSTOM_SCORE_ARTIFACT": {"result": 0},
        }


class AlternativeDummyScoreRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyScoreType, Score]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class MissingParamDummyScoreRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyScoreType, Score]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {},
        }


class InvalidParamDummyScoreRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyScoreType, Score]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {"invalid": 0},
        }


class DummyArrayType(ArtifactType):
    pass


class DummyArrayRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyArrayType, Array]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotType(ArtifactType):
    pass


class DummyPlotRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyPlotType, Plot]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyScoreCollectionType(ArtifactType):
    pass


class DummyScoreCollectionRegistry(
    ArtifactRegistry[
        DummyArtifactResources, DummyResourceSpec, DummyScoreCollectionType, ScoreCollection
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyArrayCollectionType(ArtifactType):
    pass


class DummyArrayCollectionRegistry(
    ArtifactRegistry[
        DummyArtifactResources, DummyResourceSpec, DummyArrayCollectionType, ArrayCollection
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotCollectionType(ArtifactType):
    pass


class DummyPlotCollectionRegistry(
    ArtifactRegistry[
        DummyArtifactResources, DummyResourceSpec, DummyPlotCollectionType, PlotCollection
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
