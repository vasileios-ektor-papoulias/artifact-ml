from typing import Any, Dict

from artifact_core._base.registry import (
    ArtifactRegistry,
    ArtifactType,
)
from matplotlib.figure import Figure
from numpy import ndarray

from tests.base.dummy.artifact_dependencies import DummyArtifactResources, DummyResourceSpec


class DummyScoreType(ArtifactType):
    DUMMY_SCORE_ARTIFACT = "dummy_score_artifact"
    NO_HYPERPARAMS_ARTIFACT = "no_hyperparams_artifact"
    IN_ALTERNATIVE_REGISTRY = "in_alternative_registry"
    NOT_REGISTERED = "not_registered"


class DummyScoreRegistry(
    ArtifactRegistry[DummyScoreType, DummyArtifactResources, float, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {"adjust_scale": True},
            "CUSTOM_SCORE_ARTIFACT": {"result": 0},
        }


class AlternativeDummyScoreRegistry(
    ArtifactRegistry[DummyScoreType, DummyArtifactResources, float, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class MissingParamDummyScoreRegistry(
    ArtifactRegistry[DummyScoreType, DummyArtifactResources, float, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {},
        }


class InvalidParamDummyScoreRegistry(
    ArtifactRegistry[DummyScoreType, DummyArtifactResources, float, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {"invalid": 0},
        }


class DummyArrayType(ArtifactType):
    pass


class DummyArrayRegistry(
    ArtifactRegistry[DummyArrayType, DummyArtifactResources, ndarray, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotType(ArtifactType):
    pass


class DummyPlotRegistry(
    ArtifactRegistry[DummyPlotType, DummyArtifactResources, Figure, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyScoreCollectionType(ArtifactType):
    pass


class DummyScoreCollectionRegistry(
    ArtifactRegistry[
        DummyScoreCollectionType, DummyArtifactResources, Dict[str, float], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyArrayCollectionType(ArtifactType):
    pass


class DummyArrayCollectionRegistry(
    ArtifactRegistry[
        DummyArrayCollectionType, DummyArtifactResources, Dict[str, ndarray], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotCollectionType(ArtifactType):
    pass


class DummyPlotCollectionRegistry(
    ArtifactRegistry[
        DummyPlotCollectionType, DummyArtifactResources, Dict[str, Figure], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
