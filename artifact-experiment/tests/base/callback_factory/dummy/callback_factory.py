from typing import Type

from artifact_experiment.base.plans.callback_factory import ArtifactCallbackFactory

from tests.base.dummy_artifact_toolkit import (
    DummyArray,
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyArtifactResources,
    DummyPlot,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
    DummyResourceSpec,
    DummyScoreCollectionRegistry,
    DummyScoreCollectionType,
    DummyScoreRegistry,
    DummyScoreType,
)


class DummyCallbackFactory(
    ArtifactCallbackFactory[
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
