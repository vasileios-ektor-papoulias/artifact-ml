from typing import Type

from artifact_core._base.orchestration.engine import ArtifactEngine

from tests._base.dummy.registries import (
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
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources


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
    def _get_score_registry(cls) -> Type[DummyScoreRegistry]:
        return DummyScoreRegistry

    @classmethod
    def _get_array_registry(cls) -> Type[DummyArrayRegistry]:
        return DummyArrayRegistry

    @classmethod
    def _get_plot_registry(cls):
        return DummyPlotRegistry

    @classmethod
    def _get_score_collection_registry(cls) -> Type[DummyScoreCollectionRegistry]:
        return DummyScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(cls) -> Type[DummyArrayCollectionRegistry]:
        return DummyArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(cls) -> Type[DummyPlotCollectionRegistry]:
        return DummyPlotCollectionRegistry
