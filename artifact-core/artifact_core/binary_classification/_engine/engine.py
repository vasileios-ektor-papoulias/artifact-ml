from typing import Type

from artifact_core.binary_classification._engine.base import (
    BinaryClassificationEngineBase,
)
from artifact_core.binary_classification._registries.array_collections.registry import (
    BinaryClassificationArrayCollectionRegistry,
    BinaryClassificationArrayCollectionType,
)
from artifact_core.binary_classification._registries.arrays.registry import (
    BinaryClassificationArrayRegistry,
    BinaryClassificationArrayType,
)
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationArrayCollectionRegistryBase,
    BinaryClassificationArrayRegistryBase,
    BinaryClassificationPlotCollectionRegistryBase,
    BinaryClassificationPlotRegistryBase,
    BinaryClassificationScoreCollectionRegistryBase,
    BinaryClassificationScoreRegistryBase,
)
from artifact_core.binary_classification._registries.plot_collections.registry import (
    BinaryClassificationPlotCollectionRegistry,
    BinaryClassificationPlotCollectionType,
)
from artifact_core.binary_classification._registries.plots.registry import (
    BinaryClassificationPlotRegistry,
    BinaryClassificationPlotType,
)
from artifact_core.binary_classification._registries.score_collections.registry import (
    BinaryClassificationScoreCollectionRegistry,
    BinaryClassificationScoreCollectionType,
)
from artifact_core.binary_classification._registries.scores.registry import (
    BinaryClassificationScoreRegistry,
    BinaryClassificationScoreType,
)


class BinaryClassificationEngine(
    BinaryClassificationEngineBase[
        BinaryClassificationScoreType,
        BinaryClassificationArrayType,
        BinaryClassificationPlotType,
        BinaryClassificationScoreCollectionType,
        BinaryClassificationArrayCollectionType,
        BinaryClassificationPlotCollectionType,
    ]
):
    @classmethod
    def _get_score_registry(
        cls,
    ) -> Type[BinaryClassificationScoreRegistryBase]:
        return BinaryClassificationScoreRegistry

    @classmethod
    def _get_array_registry(
        cls,
    ) -> Type[BinaryClassificationArrayRegistryBase]:
        return BinaryClassificationArrayRegistry

    @classmethod
    def _get_plot_registry(
        cls,
    ) -> Type[BinaryClassificationPlotRegistryBase]:
        return BinaryClassificationPlotRegistry

    @classmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[BinaryClassificationScoreCollectionRegistryBase]:
        return BinaryClassificationScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[BinaryClassificationArrayCollectionRegistryBase]:
        return BinaryClassificationArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[BinaryClassificationPlotCollectionRegistryBase]:
        return BinaryClassificationPlotCollectionRegistry
