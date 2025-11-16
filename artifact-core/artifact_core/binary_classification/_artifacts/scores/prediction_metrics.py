from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.prediction_metrics.calculator import (
    BinaryPredictionMetric,
    BinaryPredictionMetricCalculator,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScore
from artifact_core.binary_classification._registries.scores import BinaryClassificationScoreRegistry
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.ACCURACY)
class AccuracyScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.ACCURACY,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.BALANCED_ACCURACY
)
class BalancedAccuracyScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.BALANCED_ACCURACY,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.PRECISION)
class PrecisionScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.PRECISION,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.NPV)
class NPVScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.NPV,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.RECALL)
class RecallScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.RECALL,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.TNR)
class TNRScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.TNR,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.FPR)
class FPRScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.FPR,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.FNR)
class FNRScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.FNR,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.F1)
class F1ScoreScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.F1,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result


@BinaryClassificationScoreRegistry.register_artifact(BinaryClassificationScoreType.MCC)
class MCCScore(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        result = BinaryPredictionMetricCalculator.compute(
            metric_type=BinaryPredictionMetric.MCC,
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
        )
        return result
