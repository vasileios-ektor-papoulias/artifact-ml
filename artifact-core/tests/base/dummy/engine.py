from artifact_core._base.engine import ArtifactEngine

from tests.base.dummy.artifact_dependencies import DummyArtifactResources, DummyResourceSpec
from tests.base.dummy.registries import (
    DummyArray,
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyPlot,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
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
        DummyArray,
        DummyPlot,
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
