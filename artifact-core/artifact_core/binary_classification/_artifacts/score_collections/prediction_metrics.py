from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.prediction_metrics.calculator import (
    BinaryPredictionMetric,
    BinaryPredictionMetricCalculator,
    BinaryPredictionMetricLiteral,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScoreCollection
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)

BinaryPredictionScoresHyperparamsT = TypeVar(
    "BinaryPredictionScoresHyperparamsT", bound="BinaryPredictionScoresHyperparams"
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.BINARY_PREDICTION_SCORES
)
@dataclass(frozen=True)
class BinaryPredictionScoresHyperparams(ArtifactHyperparams):
    metric_types: Sequence[BinaryPredictionMetric]

    @classmethod
    def build(
        cls: Type[BinaryPredictionScoresHyperparamsT],
        metric_types: Sequence[Union[BinaryPredictionMetric, BinaryPredictionMetricLiteral]],
    ) -> BinaryPredictionScoresHyperparamsT:
        ls_resolved = [
            metric_type
            if isinstance(metric_type, BinaryPredictionMetric)
            else BinaryPredictionMetric[metric_type]
            for metric_type in metric_types
        ]
        hyperparams = cls(metric_types=ls_resolved)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.BINARY_PREDICTION_SCORES
)
class BinaryPredictionScores(
    BinaryClassificationScoreCollection[BinaryPredictionScoresHyperparams]
):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_score_collection = BinaryPredictionMetricCalculator.compute_multiple(
            metric_types=self._hyperparams.metric_types,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        result = {
            metric_type.value: metric_value
            for metric_type, metric_value in dict_score_collection.items()
        }
        return result
