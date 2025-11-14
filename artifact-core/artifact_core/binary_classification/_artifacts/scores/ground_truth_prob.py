from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.classification.ground_truth_prob.stats_calculator import (
    DescriptiveStatistic,
    GroundTruthProbStatsCalculator,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScore
from artifact_core.binary_classification._registries.scores import BinaryClassificationScoreRegistry
from artifact_core.binary_classification._types.scores import BinaryClassificationScoreType


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MEAN
)
class GroundTruthProbMean(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.MEAN,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_STD
)
class GroundTruthProbSTD(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.STD,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_VARIANCE
)
class GroundTruthProbVariance(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.VARIANCE,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MEDIAN
)
class GroundTruthProbMedian(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.MEDIAN,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_FIRST_QUARTILE
)
class GroundTruthProbFirstQuartile(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.Q1,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_THIRD_QUARTILE
)
class GroundTruthProbThirdQuartile(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.Q3,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MIN
)
class GroundTruthProbMin(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.MIN,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MAX
)
class GroundTruthProbMax(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthProbStatsCalculator.compute(
            stat=DescriptiveStatistic.MAX,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return score
