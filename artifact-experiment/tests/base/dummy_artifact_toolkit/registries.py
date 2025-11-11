from typing import Any, Dict

from artifact_core._base.registry import (
    ArtifactRegistry,
    ArtifactType,
)
from matplotlib.figure import Figure
from numpy import ndarray

from tests.base.dummy_artifact_toolkit.artifact_dependencies import (
    DummyArtifactResources,
    DummyResourceSpec,
)


class DummyScoreType(ArtifactType):
    DUMMY_SCORE_1 = "dummy_score_1"


class DummyScoreRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyScoreType, float]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyArrayType(ArtifactType):
    DUMMY_ARRAY_1 = "dummy_array_1"


class DummyArrayRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyArrayType, ndarray]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotType(ArtifactType):
    DUMMY_PLOT_1 = "dummy_plot_1"


class DummyPlotRegistry(
    ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyPlotType, Figure]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyScoreCollectionType(ArtifactType):
    DUMMY_SCORE_COLLECTION_1 = "dummy_score_collection_1"


class DummyScoreCollectionRegistry(
    ArtifactRegistry[
        DummyArtifactResources, DummyResourceSpec, DummyScoreCollectionType, Dict[str, float]
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyArrayCollectionType(ArtifactType):
    DUMMY_ARRAY_COLLECTION_1 = "dummy_array_collection_1"


class DummyArrayCollectionRegistry(
    ArtifactRegistry[
        DummyArtifactResources,
        DummyResourceSpec,
        DummyArrayCollectionType,
        Dict[str, ndarray],
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyPlotCollectionType(ArtifactType):
    DUMMY_PLOT_COLLECTION_1 = "dummy_plot_collection_1"


class DummyPlotCollectionRegistry(
    ArtifactRegistry[
        DummyArtifactResources,
        DummyResourceSpec,
        DummyPlotCollectionType,
        Dict[str, Figure],
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
