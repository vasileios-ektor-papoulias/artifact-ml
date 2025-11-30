from typing import Type

from artifact_core._base.orchestration.registry import ArtifactRegistry
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from tests._base.dummy.engine.base import DummyArtifactEngineBase
from tests._base.dummy.registries.array_collections import DummyArrayCollectionRegistry
from tests._base.dummy.registries.arrays import DummyArrayRegistry
from tests._base.dummy.registries.plot_collections import DummyPlotCollectionRegistry
from tests._base.dummy.registries.plots import DummyPlotRegistry
from tests._base.dummy.registries.score_collections import DummyScoreCollectionRegistry
from tests._base.dummy.registries.scores import DummyScoreRegistry
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources
from tests._base.dummy.types.array_collections import DummyArrayCollectionType
from tests._base.dummy.types.arrays import DummyArrayType
from tests._base.dummy.types.plot_collections import DummyPlotCollectionType
from tests._base.dummy.types.plots import DummyPlotType
from tests._base.dummy.types.score_collections import DummyScoreCollectionType
from tests._base.dummy.types.scores import DummyScoreType


class DummyArtifactEngine(
    DummyArtifactEngineBase[
        DummyScoreType,
        DummyArrayType,
        DummyPlotType,
        DummyScoreCollectionType,
        DummyArrayCollectionType,
        DummyPlotCollectionType,
    ]
):
    @classmethod
    def _get_score_registry(
        cls,
    ) -> Type[ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyScoreType, Score]]:
        return DummyScoreRegistry

    @classmethod
    def _get_array_registry(
        cls,
    ) -> Type[ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyArrayType, Array]]:
        return DummyArrayRegistry

    @classmethod
    def _get_plot_registry(
        cls,
    ) -> Type[ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyPlotType, Plot]]:
        return DummyPlotRegistry

    @classmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            DummyArtifactResources, DummyResourceSpec, DummyScoreCollectionType, ScoreCollection
        ]
    ]:
        return DummyScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            DummyArtifactResources, DummyResourceSpec, DummyArrayCollectionType, ArrayCollection
        ]
    ]:
        return DummyArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[
        ArtifactRegistry[
            DummyArtifactResources, DummyResourceSpec, DummyPlotCollectionType, PlotCollection
        ]
    ]:
        return DummyPlotCollectionRegistry
