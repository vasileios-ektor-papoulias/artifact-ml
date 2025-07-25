from typing import Type

from artifact_experiment.base.callbacks.factory import ArtifactCallbackFactory

from tests.base.dummy.registries import (
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyArrayType,
    DummyArtifactResources,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
    DummyPlotType,
    DummyResourceSpec,
    DummyScoreCollectionRegistry,
    DummyScoreCollectionType,
    DummyScoreRegistry,
    DummyScoreType,
)


class TableComparisonCallbackFactory(
    ArtifactCallbackFactory[
        DummyScoreType,
        DummyArrayType,
        DummyPlotType,
        DummyScoreCollectionType,
        DummyArrayCollectionType,
        DummyPlotCollectionType,
        DummyArtifactResources,
        DummyResourceSpec,
    ]
):
    @staticmethod
    def _get_score_registry() -> Type[DummyScoreRegistry]:
        return DummyScoreRegistry

    @staticmethod
    def _get_array_registry() -> Type[DummyArrayRegistry]:
        return DummyArrayRegistry

    @staticmethod
    def _get_plot_registry() -> Type[DummyPlotRegistry]:
        return DummyPlotRegistry

    @staticmethod
    def _get_score_collection_registry() -> Type[DummyScoreCollectionRegistry]:
        return DummyScoreCollectionRegistry

    @staticmethod
    def _get_array_collection_registry() -> Type[DummyArrayCollectionRegistry]:
        return DummyArrayCollectionRegistry

    @staticmethod
    def _get_plot_collection_registry() -> Type[DummyPlotCollectionRegistry]:
        return DummyPlotCollectionRegistry
