from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationScoreCollection,
)
from artifact_core.binary_classification.registries.score_collections.registry import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification.registries.score_collections.types import (
    BinaryClassificationScoreCollectionType,
)
from artifact_core.libs.implementation.binary_classification.prediction_metrics.calculator import (
    BinaryPredictionMetric,
    BinaryPredictionMetricCalculator,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)

BinaryPredictionMetricLiteral = Literal[
    "ACCURACY",
    "BALANCED_ACCURACY",
    "PRECISION",
    "NPV",
    "RECALL",
    "TNR",
    "FPR",
    "FNR",
    "F1",
    "MCC",
]

BinaryPredictionScoresHyperparamsT = TypeVar(
    "BinaryPredictionScoresHyperparamsT",
    bound="BinaryPredictionScoresHyperparams",
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_config(
    BinaryClassificationScoreCollectionType.BINARY_PREDICTION_SCORES
)
@dataclass(frozen=True)
class BinaryPredictionScoresHyperparams(ArtifactHyperparams):
    ls_metrics: List[BinaryPredictionMetric]

    @classmethod
    def build(
        cls: Type[BinaryPredictionScoresHyperparamsT],
        ls_metrics: Sequence[Union[BinaryPredictionMetric, BinaryPredictionMetricLiteral]],
    ) -> BinaryPredictionScoresHyperparamsT:
        ls_resolved = [
            metric_type
            if isinstance(metric_type, BinaryPredictionMetric)
            else BinaryPredictionMetric[metric_type]
            for metric_type in ls_metrics
        ]
        hyperparams = cls(ls_metrics=ls_resolved)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.BINARY_PREDICTION_SCORES
)
class BinaryPredictionScores(
    BinaryClassificationScoreCollection[BinaryPredictionScoresHyperparams]
):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_score_collection = BinaryPredictionMetricCalculator.compute_multiple(
            metrics=self._hyperparams.ls_metrics,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        result = {
            metric_type.value: metric_value
            for metric_type, metric_value in dict_score_collection.items()
        }
        return result
