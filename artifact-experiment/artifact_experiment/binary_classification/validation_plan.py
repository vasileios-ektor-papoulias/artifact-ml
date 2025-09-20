from abc import abstractmethod
from typing import List, Type

from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationArtifactResources,
    BinaryFeatureSpecProtocol,
)
from artifact_core.libs.resources.classification.classification_results import (
    BinaryClassificationResults,
)

from artifact_experiment.base.callback_factory import ArtifactCallbackFactory
from artifact_experiment.base.validation_plan import ValidationPlan
from artifact_experiment.binary_classification.callback_factory import (
    BinaryClassificationArrayCollectionType,
    BinaryClassificationArrayType,
    BinaryClassificationCallbackFactory,
    BinaryClassificationPlotCollectionType,
    BinaryClassificationPlotType,
    BinaryClassificationScoreCollectionType,
    BinaryClassificationScoreType,
)
from artifact_experiment.binary_classification.resources import (
    BinaryClassificationCallbackResources,
)


class BinaryClassifierEvaluationPlan(
    ValidationPlan[
        BinaryClassificationScoreType,
        BinaryClassificationArrayType,
        BinaryClassificationPlotType,
        BinaryClassificationScoreCollectionType,
        BinaryClassificationArrayCollectionType,
        BinaryClassificationPlotCollectionType,
        BinaryClassificationArtifactResources,
        BinaryFeatureSpecProtocol,
    ]
):
    @staticmethod
    @abstractmethod
    def _get_score_types() -> List[BinaryClassificationScoreType]: ...

    @staticmethod
    @abstractmethod
    def _get_array_types() -> List[BinaryClassificationArrayType]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_types() -> List[BinaryClassificationPlotType]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_types() -> List[BinaryClassificationScoreCollectionType]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_types() -> List[BinaryClassificationArrayCollectionType]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_types() -> List[BinaryClassificationPlotCollectionType]: ...

    def execute_table_comparison(self, classification_results: BinaryClassificationResults):
        callback_resources = BinaryClassificationCallbackResources.build(
            classification_results=classification_results
        )
        super().execute(resources=callback_resources)

    @staticmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            BinaryClassificationScoreType,
            BinaryClassificationArrayType,
            BinaryClassificationPlotType,
            BinaryClassificationScoreCollectionType,
            BinaryClassificationArrayCollectionType,
            BinaryClassificationPlotCollectionType,
            BinaryClassificationArtifactResources,
            BinaryFeatureSpecProtocol,
        ]
    ]:
        return BinaryClassificationCallbackFactory
