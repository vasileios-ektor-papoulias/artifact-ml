from typing import Type

from artifact_core.binary_classification._engine.base import BinaryClassificationEngineBase
from artifact_core.binary_classification._registries.array_collections import (
    BinaryClassificationArrayCollectionRegistry,
)
from artifact_core.binary_classification._registries.arrays import BinaryClassificationArrayRegistry
from artifact_core.binary_classification._registries.base import (
    BinaryClassificationArrayCollectionRegistryBase,
    BinaryClassificationArrayRegistryBase,
    BinaryClassificationPlotCollectionRegistryBase,
    BinaryClassificationPlotRegistryBase,
    BinaryClassificationScoreCollectionRegistryBase,
    BinaryClassificationScoreRegistryBase,
)
from artifact_core.binary_classification._registries.plot_collections import (
    BinaryClassificationPlotCollectionRegistry,
)
from artifact_core.binary_classification._registries.plots import BinaryClassificationPlotRegistry
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._registries.scores import BinaryClassificationScoreRegistry
from artifact_core.binary_classification._types.array_collections import (
    BinaryClassificationArrayCollectionType,
)
from artifact_core.binary_classification._types.arrays import BinaryClassificationArrayType
from artifact_core.binary_classification._types.plot_collections import (
    BinaryClassificationPlotCollectionType,
)
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType


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
    ) -> Type[BinaryClassificationScoreRegistryBase[BinaryClassificationScoreType]]:
        return BinaryClassificationScoreRegistry

    @classmethod
    def _get_array_registry(
        cls,
    ) -> Type[BinaryClassificationArrayRegistryBase[BinaryClassificationArrayType]]:
        return BinaryClassificationArrayRegistry

    @classmethod
    def _get_plot_registry(
        cls,
    ) -> Type[BinaryClassificationPlotRegistryBase[BinaryClassificationPlotType]]:
        return BinaryClassificationPlotRegistry

    @classmethod
    def _get_score_collection_registry(
        cls,
    ) -> Type[
        BinaryClassificationScoreCollectionRegistryBase[BinaryClassificationScoreCollectionType]
    ]:
        return BinaryClassificationScoreCollectionRegistry

    @classmethod
    def _get_array_collection_registry(
        cls,
    ) -> Type[
        BinaryClassificationArrayCollectionRegistryBase[BinaryClassificationArrayCollectionType]
    ]:
        return BinaryClassificationArrayCollectionRegistry

    @classmethod
    def _get_plot_collection_registry(
        cls,
    ) -> Type[
        BinaryClassificationPlotCollectionRegistryBase[BinaryClassificationPlotCollectionType]
    ]:
        return BinaryClassificationPlotCollectionRegistry
