from typing import Any, Dict

from artifact_core.base.registry import (
    ArtifactRegistry,
    ArtifactType,
)
from matplotlib.figure import Figure
from numpy import ndarray

from tests.base.dummy.artifact_dependencies import DummyArtifactResources, DummyResourceSpec


class DummyScoreType(ArtifactType):
    DUMMY_SCORE_ARTIFACT = "dummy_score_artifact"


class DummyScoreRegistry(
    ArtifactRegistry[DummyScoreType, DummyArtifactResources, float, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyArrayType(ArtifactType):
    DUMMY_ARRAY_ARTIFACT = "dummy_array_artifact"


class DummyArrayRegistry(
    ArtifactRegistry[DummyArrayType, DummyArtifactResources, ndarray, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotType(ArtifactType):
    DUMMY_PLOT_ARTIFACT = "dummy_plot_artifact"


class DummyPlotRegistry(
    ArtifactRegistry[DummyPlotType, DummyArtifactResources, Figure, DummyResourceSpec]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyScoreCollectionType(ArtifactType):
    DUMMY_SCORE_COLLECTION_ARTIFACT = "dummy_score_collection_artifact"


class DummyScoreCollectionRegistry(
    ArtifactRegistry[
        DummyScoreCollectionType, DummyArtifactResources, Dict[str, float], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyArrayCollectionType(ArtifactType):
    DUMMY_ARRAY_COLLECTION_ARTIFACT = "dummy_array_collection_artifact"


class DummyArrayCollectionRegistry(
    ArtifactRegistry[
        DummyArrayCollectionType, DummyArtifactResources, Dict[str, ndarray], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotCollectionType(ArtifactType):
    DUMMY_PLOT_COLLECTION_ARTIFACT = "dummy_plot_collection_artifact"


class DummyPlotCollectionRegistry(
    ArtifactRegistry[
        DummyPlotCollectionType, DummyArtifactResources, Dict[str, Figure], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
