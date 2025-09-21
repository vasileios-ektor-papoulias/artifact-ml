from abc import abstractmethod
from typing import Dict, List, Optional, Type

from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationArtifactResources,
    BinaryFeatureSpecProtocol,
)
from artifact_core.libs.types.entity_store import IdentifierType
from numpy import ndarray

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
from artifact_experiment.binary_classification.callback_resources import (
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

    def execute_classifier_evaluation(
        self,
        true: Dict[IdentifierType, str],
        predicted: Dict[IdentifierType, str],
        logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ):
        callback_resources = BinaryClassificationCallbackResources.from_spec(
            class_spec=self._resource_spec,
            true=true,
            predicted=predicted,
            logits=logits,
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
