from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from artifact_core._base.artifact_dependencies import ArtifactHyperparams
from artifact_core._libs.implementation.binary_classification.threshold_variation.calculator import (
    ThresholdVariationMetric,
    ThresholdVariationMetricCalculator,
    ThresholdVariationMetricLiteral,
)
from artifact_core._libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core._libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import (
    BinaryClassificationScoreCollection,
)
from artifact_core.binary_classification._registries.score_collections.registry import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._registries.score_collections.types import (
    BinaryClassificationScoreCollectionType,
)

ThresholdVariationScoresHyperparamsT = TypeVar(
    "ThresholdVariationScoresHyperparamsT", bound="ThresholdVariationScoresHyperparams"
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.THRESHOLD_VARIATION_SCORES
)
@dataclass(frozen=True)
class ThresholdVariationScoresHyperparams(ArtifactHyperparams):
    metric_types: Sequence[ThresholdVariationMetric]

    @classmethod
    def build(
        cls: Type[ThresholdVariationScoresHyperparamsT],
        metric_types: Sequence[Union[ThresholdVariationMetric, ThresholdVariationMetricLiteral]],
    ) -> ThresholdVariationScoresHyperparamsT:
        ls_resolved = [
            metric_type
            if isinstance(metric_type, ThresholdVariationMetric)
            else ThresholdVariationMetric[metric_type]
            for metric_type in metric_types
        ]
        hyperparams = cls(metric_types=ls_resolved)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.THRESHOLD_VARIATION_SCORES
)
class ThresholdVariationScores(
    BinaryClassificationScoreCollection[ThresholdVariationScoresHyperparams]
):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_score_collection = ThresholdVariationMetricCalculator.compute_multiple(
            metric_types=self._hyperparams.metric_types,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )
        result = {
            metric_type.value: metric_value
            for metric_type, metric_value in dict_score_collection.items()
        }
        return result
