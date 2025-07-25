from typing import List, Type

from artifact_experiment.base.validation_plan import ValidationPlan

from tests.base.callbacks.dummy.factory import DummyCallbackFactory
from tests.base.dummy import (
    DummyArrayCollectionType,
    DummyArrayType,
    DummyArtifactResources,
    DummyPlotCollectionType,
    DummyPlotType,
    DummyResourceSpec,
    DummyScoreCollectionType,
    DummyScoreType,
)


class DummyValidationPlan(
    ValidationPlan[
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
    def _get_score_types() -> List[DummyScoreType]:
        return [DummyScoreType.DUMMY_SCORE_1]

    @staticmethod
    def _get_array_types() -> List[DummyArrayType]:
        return [DummyArrayType.DUMMY_ARRAY_1]

    @staticmethod
    def _get_plot_types() -> List[DummyPlotType]:
        return [DummyPlotType.DUMMY_PLOT_1]

    @staticmethod
    def _get_score_collection_types() -> List[DummyScoreCollectionType]:
        return [DummyScoreCollectionType.DUMMY_SCORE_COLLECTION_1]

    @staticmethod
    def _get_array_collection_types() -> List[DummyArrayCollectionType]:
        return [DummyArrayCollectionType.DUMMY_ARRAY_COLLECTION_1]

    @staticmethod
    def _get_plot_collection_types() -> List[DummyPlotCollectionType]:
        return [DummyPlotCollectionType.DUMMY_PLOT_COLLECTION_1]

    @staticmethod
    def _get_callback_factory() -> Type[DummyCallbackFactory]:
        return DummyCallbackFactory
