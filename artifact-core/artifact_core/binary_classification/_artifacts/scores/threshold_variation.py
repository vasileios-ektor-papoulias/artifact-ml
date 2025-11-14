from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.threshold_variation.calculator import (
    ThresholdVariationMetric,
    ThresholdVariationMetricCalculator,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScore
from artifact_core.binary_classification._registries.scores import BinaryClassificationScoreRegistry
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.ROC_AUC)
class ROCAUCScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = ThresholdVariationMetricCalculator.compute(
            metric_type=ThresholdVariationMetric.ROC_AUC,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.PR_AUC)
class PRAUCScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = ThresholdVariationMetricCalculator.compute(
            metric_type=ThresholdVariationMetric.PR_AUC,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_predicted_positive,
        )
        return result
