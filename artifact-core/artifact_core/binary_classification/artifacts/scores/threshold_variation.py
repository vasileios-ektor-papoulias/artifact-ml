from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import BinaryClassificationScore
from artifact_core.binary_classification.registries.scores.registry import (
    BinaryClassificationScoreRegistry,
)
from artifact_core.binary_classification.registries.scores.types import (
    BinaryClassificationScoreType,
)
from artifact_core.libs.implementation.binary_classification.threshold_variation.calculator import (
    ThresholdVariationMetric,
    ThresholdVariationMetricCalculator,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.ROC_AUC)
class ROCAUCScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = ThresholdVariationMetricCalculator.compute(
            metric_type=ThresholdVariationMetric.ROC_AUC,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.PR_AUC)
class PRAUCScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = ThresholdVariationMetricCalculator.compute(
            metric_type=ThresholdVariationMetric.PR_AUC,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_predicted_positive,
        )
        return result
