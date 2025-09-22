from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import BinaryClassificationScore
from artifact_core.binary_classification.registries.scores.registry import (
    BinaryClassificationScoreRegistry,
)
from artifact_core.binary_classification.registries.scores.types import (
    BinaryClassificationScoreType,
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


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.ACCURACY)
class AccuracyScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.ACCURACY,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.BALANCED_ACCURACY
)
class BalancedAccuracyScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.BALANCED_ACCURACY,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.PRECISION)
class PrecisionScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.PRECISION,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.NPV)
class NPVScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.NPV,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.RECALL)
class RecallScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.RECALL,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.TNR)
class TNRScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.TNR,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.FPR)
class FPRScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.FPR,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.FNR)
class FNRScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.FNR,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.F1)
class F1ScoreScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.F1,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.MCC)
class MCCScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.MCC,
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return result
