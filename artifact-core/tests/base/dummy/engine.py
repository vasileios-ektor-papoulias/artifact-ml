from artifact_core.base.engine import ArtifactEngine

from tests.base.dummy.artifact_dependencies import DummyArtifactResources, DummyResourceSpec
from tests.base.dummy.registries import (
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyArrayType,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
    DummyPlotType,
    DummyScoreCollectionRegistry,
    DummyScoreCollectionType,
    DummyScoreRegistry,
    DummyScoreType,
)


class DummyArtifactEngine(
    ArtifactEngine[
        DummyArtifactResources,
        DummyResourceSpec,
        DummyScoreType,
        DummyArrayType,
        DummyPlotType,
        DummyScoreCollectionType,
        DummyArrayCollectionType,
        DummyPlotCollectionType,
    ]
):
    @classmethod
    def _get_score_registry(cls):
        return DummyScoreRegistry

    @classmethod
    def _get_array_registry(cls):
        return DummyArrayRegistry

    @classmethod
    def _get_plot_registry(cls):
        return DummyPlotRegistry

    @classmethod
    def _get_score_collection_registry(cls):
        return DummyScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(cls):
        return DummyArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(cls):
        return DummyPlotCollectionRegistry
