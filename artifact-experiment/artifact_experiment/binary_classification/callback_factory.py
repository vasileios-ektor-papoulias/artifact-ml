from typing import Type

from artifact_core.binary_classification._artifacts.base import (
    BinaryClassificationArtifactResources,
    BinaryFeatureSpecProtocol,
)
from artifact_core.binary_classification._registries.array_collections.registry import (
    BinaryClassificationArrayCollectionRegistry,
    BinaryClassificationArrayCollectionRegistryBase,
    BinaryClassificationArrayCollectionType,
)
from artifact_core.binary_classification._registries.arrays.registry import (
    BinaryClassificationArrayRegistry,
    BinaryClassificationArrayRegistryBase,
    BinaryClassificationArrayType,
)
from artifact_core.binary_classification._registries.plot_collections.registry import (
    BinaryClassificationPlotCollectionRegistry,
    BinaryClassificationPlotCollectionRegistryBase,
    BinaryClassificationPlotCollectionType,
)
from artifact_core.binary_classification._registries.plots.registry import (
    BinaryClassificationPlotRegistry,
    BinaryClassificationPlotRegistryBase,
    BinaryClassificationPlotType,
)
from artifact_core.binary_classification._registries.score_collections.registry import (
    BinaryClassificationScoreCollectionRegistry,
    BinaryClassificationScoreCollectionRegistryBase,
    BinaryClassificationScoreCollectionType,
)
from artifact_core.binary_classification._registries.scores.registry import (
    BinaryClassificationScoreRegistry,
    BinaryClassificationScoreRegistryBase,
    BinaryClassificationScoreType,
)

from artifact_experiment.base.components.factories.artifact import ArtifactCallbackFactory


class BinaryClassificationCallbackFactory(
    ArtifactCallbackFactory[
        BinaryClassificationArtifactResources,
        BinaryFeatureSpecProtocol,
        BinaryClassificationScoreType,
        BinaryClassificationArrayType,
        BinaryClassificationPlotType,
        BinaryClassificationScoreCollectionType,
        BinaryClassificationArrayCollectionType,
        BinaryClassificationPlotCollectionType,
    ]
):
    @staticmethod
    def _get_score_registry() -> Type[
        BinaryClassificationScoreRegistryBase[BinaryClassificationScoreType]
    ]:
        return BinaryClassificationScoreRegistry

    @staticmethod
    def _get_array_registry() -> Type[
        BinaryClassificationArrayRegistryBase[BinaryClassificationArrayType]
    ]:
        return BinaryClassificationArrayRegistry

    @staticmethod
    def _get_plot_registry() -> Type[
        BinaryClassificationPlotRegistryBase[BinaryClassificationPlotType]
    ]:
        return BinaryClassificationPlotRegistry

    @staticmethod
    def _get_score_collection_registry() -> Type[
        BinaryClassificationScoreCollectionRegistryBase[BinaryClassificationScoreCollectionType]
    ]:
        return BinaryClassificationScoreCollectionRegistry

    @staticmethod
    def _get_array_collection_registry() -> Type[
        BinaryClassificationArrayCollectionRegistryBase[BinaryClassificationArrayCollectionType]
    ]:
        return BinaryClassificationArrayCollectionRegistry

    @staticmethod
    def _get_plot_collection_registry() -> Type[
        BinaryClassificationPlotCollectionRegistryBase[BinaryClassificationPlotCollectionType]
    ]:
        return BinaryClassificationPlotCollectionRegistry
