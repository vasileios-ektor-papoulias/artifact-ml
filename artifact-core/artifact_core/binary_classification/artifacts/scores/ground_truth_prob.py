from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import BinaryClassificationScore
from artifact_core.binary_classification.registries.scores.registry import (
    BinaryClassificationScoreRegistry,
)
from artifact_core.binary_classification.registries.scores.types import (
    BinaryClassificationScoreType,
)
from artifact_core.libs.implementation.classification.ground_truth_prob.stats_calculator import (
    DescriptiveStatistic,
    GroundTruthStatsCalculator,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MEAN
)
class GroundTruthProbMean(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.MEAN,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_STD
)
class GroundTruthProbSTD(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.STD,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_VARIANCE
)
class GroundTruthProbVariance(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.VARIANCE,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MEDIAN
)
class GroundTruthProbMedian(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.MEDIAN,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_FIRST_QUARTILE
)
class GroundTruthProbFirstQuartile(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.Q1,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_THIRD_QUARTILE
)
class GroundTruthProbThirdQuartile(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.Q3,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MIN
)
class GroundTruthProbMin(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.MIN,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score


@BinaryClassificationScoreRegistry.register_artifact(
    BinaryClassificationScoreType.GROUND_TRUTH_PROB_MAX
)
class GroundTruthProbMax(BinaryClassificationScore[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        score = GroundTruthStatsCalculator.compute(
            stat=DescriptiveStatistic.MAX,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return score
