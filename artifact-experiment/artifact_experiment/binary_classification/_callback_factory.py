from typing import Type

from artifact_core.binary_classification import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
)
from artifact_core.binary_classification.spi import (
    BinaryClassificationArrayCollectionRegistry,
    BinaryClassificationArrayRegistry,
    BinaryClassificationArtifactResources,
    BinaryClassificationPlotCollectionRegistry,
    BinaryClassificationPlotRegistry,
    BinaryClassificationScoreCollectionRegistry,
    BinaryClassificationScoreRegistry,
    BinaryClassSpecProtocol,
)

from artifact_experiment._base.components.factories.artifact import ArtifactCallbackFactory


class BinaryClassificationCallbackFactory(
    ArtifactCallbackFactory[
        BinaryClassificationArtifactResources,
        BinaryClassSpecProtocol,
        BinaryClassificationScoreType,
        BinaryClassificationArrayType,
        BinaryClassificationPlotType,
        BinaryClassificationScoreCollectionType,
        BinaryClassificationArrayCollectionType,
        BinaryClassificationPlotCollectionType,
    ]
):
    @staticmethod
    def _get_score_registry() -> Type[BinaryClassificationScoreRegistry]:
        return BinaryClassificationScoreRegistry

    @staticmethod
    def _get_array_registry() -> Type[BinaryClassificationArrayRegistry]:
        return BinaryClassificationArrayRegistry

    @staticmethod
    def _get_plot_registry() -> Type[BinaryClassificationPlotRegistry]:
        return BinaryClassificationPlotRegistry

    @staticmethod
    def _get_score_collection_registry() -> Type[BinaryClassificationScoreCollectionRegistry]:
        return BinaryClassificationScoreCollectionRegistry

    @staticmethod
    def _get_array_collection_registry() -> Type[BinaryClassificationArrayCollectionRegistry]:
        return BinaryClassificationArrayCollectionRegistry

    @staticmethod
    def _get_plot_collection_registry() -> Type[BinaryClassificationPlotCollectionRegistry]:
        return BinaryClassificationPlotCollectionRegistry
